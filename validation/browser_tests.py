# browser_tests.py — end-to-end UI tests of index.html in headless Chromium (Playwright).
# Usage: python browser_tests.py path/to/index.html path/to/test-inputs-dir   -> writes validation_browser.md
import asyncio, json, sys, os, time
from playwright.async_api import async_playwright

HTML = os.path.abspath(sys.argv[1]); FILES = os.path.abspath(sys.argv[2]); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_browser.md')
rows = []
def check(name, ok, detail=''):
    rows.append(f"| {'PASS' if ok else 'FAIL'} | {name} | {detail} |"); print(('PASS' if ok else 'FAIL'), name, detail, flush=True)

async def wait_status(pg, prefixes, timeout=900000):
    js = "(p) => { const t = document.getElementById('status').textContent; return p.some(x => t.startsWith(x)); }"
    await pg.wait_for_function(js, arg=prefixes, timeout=timeout)
    return await pg.inner_text('#status')

async def set_val(pg, id, v, ev='change'):
    await pg.evaluate("([id, v, ev]) => { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event(ev)); }", [id, str(v), ev])

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width': 1500, 'height': 1150}, accept_downloads=True)
        errors = []
        pg.on('pageerror', lambda e: errors.append(str(e)))
        pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        await pg.goto('file://' + HTML)
        await pg.wait_for_function("document.getElementById('dataInfo').textContent.includes('median')", timeout=60000)

        # baseline run on the validated region
        await pg.click('#runAll')
        st = await wait_status(pg, ['D done', 'error', 'cannot'])
        badge = await pg.inner_text('#regime')
        check('Run all (512² preset): TV and D converge, TV regime reported', 'converged' in st and 'TV regime' in badge, st.split('\n')[0] + ' / ' + badge[:60])
        log = await pg.inner_text('#log')
        check('run log lists parameters and convergence', 'ℓ_TV = 0.1' in log and 'relative gap ≤ ε' in log and 'flux-fit residual' in log, '')
        await pg.screenshot(path=os.path.join(os.path.dirname(OUT), 'screenshot_tv_regime.png'), full_page=True)

        # 1. provenance / staleness
        await set_val(pg, 'ellTv', '0.05')
        lab = await pg.inner_text('#lab_tv'); labd = await pg.inner_text('#lab_d')
        check('item 1: changing ℓ_TV marks u_TV/p and D as STALE', 'STALE' in lab and 'STALE' in labd, lab)
        before = await pg.inner_text('#lab_d')
        await pg.click('#runD'); st = await wait_status(pg, ['error', 'D done', 'cannot'])
        check('item 1: "D only" refuses to mix u,p from ℓ_TV = 0.1 with ℓ_TV = 0.05', st.startswith('error') and 'differ from the last TV run' in st and before == await pg.inner_text('#lab_d'), st[:90])
        await set_val(pg, 'ellTv', '0.1')
        lab = await pg.inner_text('#lab_tv')
        check('item 1: restoring ℓ_TV clears the STALE mark', 'STALE' not in lab, lab)
        await set_val(pg, 'ellPag', '150', 'input')
        lab = await pg.inner_text('#lab_l2'); labd = await pg.inner_text('#lab_d')
        check('item 1: changing ℓ_PAG marks u_L2 and D as STALE', 'STALE' in lab and 'STALE' in labd, lab)
        await set_val(pg, 'ellPag', '197.71', 'input')

        # 2. busy lock and discarding
        await set_val(pg, 'eps', '1e-6'); await set_val(pg, 'maxIt', '200000')
        await pg.click('#runTV'); await pg.wait_for_timeout(1500)
        dis = await pg.evaluate("['rx','ry','rw','rh','file','ellTv','presetPaper','runAll'].map(id => document.getElementById(id).disabled).concat([...document.querySelectorAll('input[name=src]')].map(e => e.disabled))")
        check('item 2: data, region, source, upload and run controls disabled while computing', all(dis), str(dis))
        roi_before = await pg.evaluate("JSON.stringify(S.roi)")
        await pg.evaluate("setROI(0, 0, 300, 300)")
        roi_after = await pg.evaluate("JSON.stringify(S.roi)")
        check('item 2: programmatic region change ignored while busy', roi_before == roi_after, roi_after)
        ver_before = await pg.evaluate("S.ver")
        await pg.click('#stop')
        st = await pg.inner_text('#status'); enabled = await pg.evaluate("!document.getElementById('rx').disabled")
        ver_after = await pg.evaluate("S.ver")
        check('item 2: Stop discards the computation (version bumped) and re-enables controls', 'stopped' in st and enabled and ver_after == ver_before + 1, st)

        # 5. presets restore everything
        await set_val(pg, 'beta', '1e-4'); await set_val(pg, 'dtol', '1e-3', 'input'); await set_val(pg, 'dmax', '7', 'input'); await set_val(pg, 'alpha', '0.5', 'input')
        await set_val(pg, 'tau', '0.2', 'input')
        await pg.click('#presetDefault'); await pg.wait_for_timeout(500)
        vals = await pg.evaluate("['ellTv','tau','eps','maxIt','ellPag','beta','alpha','ellDark','dtol','dmax','rx','ry','rw','rh'].map(id => document.getElementById(id).value)")
        check('item 5: preset restores all TV, Paganin, D and region parameters', vals == ['0.1','0.125','0.01','5000','197.71','0.01','0','1','0.000001','20000','600','700','512','512'], str(vals))

        # 9. input validation
        await set_val(pg, 'tau', '0.3', 'input'); await pg.click('#runTV'); st = await wait_status(pg, ['cannot', 'TV done', 'error'])
        check('item 9: τ ≥ 1/4 rejected before running', st.startswith('cannot') and '1/4' in st, st.replace('\n', ' ')[:80])
        await set_val(pg, 'tau', '0.125', 'input'); await set_val(pg, 'ellTv', '-1'); await pg.click('#runTV'); st = await wait_status(pg, ['cannot', 'TV done', 'error'])
        check('item 9: ℓ_TV ≤ 0 rejected', st.startswith('cannot') and 'ℓ_TV' in st, st.replace('\n', ' ')[:80])
        await set_val(pg, 'ellTv', '0.1'); await set_val(pg, 'dmax', '2.5', 'input'); await pg.click('#runD'); st = await wait_status(pg, ['cannot', 'D done', 'error'])
        check('item 9: non-integer iteration limit rejected', st.startswith('cannot'), st.replace('\n', ' ')[:80])
        await set_val(pg, 'dmax', '20000', 'input')
        await pg.check('input[name=src][value=upload]')
        await pg.set_input_files('#file', os.path.join(FILES, 'nan.npy')); await pg.wait_for_timeout(800)
        info = await pg.inner_text('#dataInfo')
        check('item 9: upload with NaN values rejected', 'non-finite' in info, info[:80])
        await pg.set_input_files('#file', os.path.join(FILES, 'neg.npy')); await pg.wait_for_timeout(800)
        info = await pg.inner_text('#dataInfo'); lab = await pg.inner_text('#lab_r')
        check('item 9: values ≤ 0 are counted and flagged (magenta), not clipped silently', 'r ≤ 0' in info and '≤ 0 (magenta)' in lab, lab)
        pix = await pg.evaluate("(() => { const c = document.getElementById('c_r').getContext('2d').getImageData(15,15,1,1).data; return [c[0],c[1],c[2]]; })()")
        check('item 9: a negative pixel is painted magenta', pix == [217, 70, 239], str(pix))

        # 3. constant image
        await pg.set_input_files('#file', os.path.join(FILES, 'const.npy')); await pg.wait_for_timeout(800)
        await pg.click('#runTV'); st = await wait_status(pg, ['TV done', 'error', 'cannot'])
        log = await pg.inner_text('#log')
        check('item 3: constant image stops at k = 1 with G = 0 (no NaN)', 'iterations 1 ' in log and 'E_p = 0' in log and 'NaN' not in log, st[:90])

        # 4. linear regime badge (full history) on the paper data
        await pg.click('#presetDefault'); await pg.wait_for_timeout(1500)
        await set_val(pg, 'ellTv', '1000'); await set_val(pg, 'maxIt', '300', 'input')
        await pg.click('#runTV'); st = await wait_status(pg, ['TV done', 'error'])
        badge = await pg.inner_text('#regime')
        check('item 4: ℓ_TV = 1000 → "linear regime throughout" (history-based)', 'Linear regime throughout' in badge and 'maximum iterations' in st, badge[:70])
        await pg.screenshot(path=os.path.join(os.path.dirname(OUT), 'screenshot_linear_regime.png'), full_page=True)

        # 10. metadata export
        await pg.click('#presetDefault'); await pg.wait_for_timeout(1500)
        await pg.click('#runAll'); await wait_status(pg, ['D done', 'error'])
        await pg.click('summary:has-text("Export")')
        async with pg.expect_download() as dl:
            await pg.click('#expMeta')
        d = await dl.value; path = await d.path(); meta = json.load(open(path))
        ok = (meta['version'] and meta['region']['width'] == 512 and meta['tv']['params']['ellTV'] == 0.1 and meta['tv']['status']['stop'] == 'relative_gap'
              and meta['D']['status']['reason'] == 'converged' and meta['D']['params']['beta'] == 0.01 and meta['data']['quantization']['bits'] == 16)
        check('item 10: metadata JSON has version, region, parameters, stopping rule and status', ok, f"file {d.suggested_filename}")
        async with pg.expect_download() as dl:
            await pg.click('button[data-exp="uTV"]')
        d2 = await dl.value
        check('item 10: array file names share the run id with the metadata', d2.suggested_filename.split('_')[1] == d.suggested_filename.split('_')[1], d2.suggested_filename)

        check('no JavaScript errors during the session', not errors, '; '.join(errors)[:200])
        await b.close()
    open(OUT, 'w').write("| result | browser test (Chromium, headless) | detail |\n|---|---|---|\n" + "\n".join(rows) + "\n")

asyncio.run(main())
