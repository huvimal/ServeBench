"""Actual model in Streamlit's app runner; no fake embedding engine."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent


def main():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
    assert not app.exception, list(app.exception)
    app.text_area[0].set_value('How do I stop automatic monthly renewals?')
    next(b for b in app.button if b.label == 'Search local documents').click().run(timeout=120)
    assert not app.exception and not app.error, (list(app.exception), list(app.error))
    rows = app.dataframe[-1].value
    assert len(rows) == 3 and {'document', 'cosine_score', 'text'} <= set(rows.columns)
    result = {'scope': 'Actual ONNX INT8 engine in Streamlit app runner; not a live-browser visual test.',
              'query': app.text_area[0].value, 'results': rows.to_dict(orient='records')}
    target = ROOT / 'reports/ui-smoke.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
