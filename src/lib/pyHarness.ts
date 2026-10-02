/**
 * Wraps user code + test code into a Python program that returns a JSON string:
 * {"ok": bool, "error": str|None, "stdout": str, "passed": int, "total": int}
 *
 * Test code convention: define functions named `test_*`; each uses `assert` with an optional message.
 */
export function buildHarness(userCode: string, tests: string): string {
  return `
import json, sys, io, traceback
_buf = io.StringIO()
_old = sys.stdout
sys.stdout = _buf
_result = {"ok": False, "error": None, "stdout": "", "passed": 0, "total": 0, "failures": []}
_ns = {}
try:
    exec(compile(${JSON.stringify(userCode)}, "solution.py", "exec"), _ns)
    exec(compile(${JSON.stringify(tests)}, "tests.py", "exec"), _ns)
    _tests = [k for k in list(_ns) if k.startswith("test_") and callable(_ns[k])]
    _result["total"] = len(_tests)
    for _t in _tests:
        try:
            _ns[_t]()
            _result["passed"] += 1
        except AssertionError as _e:
            _result["failures"].append(f"{_t}: {_e}" if str(_e) else f"{_t}: assertion failed")
        except Exception as _e:
            _result["failures"].append(f"{_t}: {type(_e).__name__}: {_e}")
    _result["ok"] = _result["total"] > 0 and _result["passed"] == _result["total"]
    if _result["failures"]:
        _result["error"] = "\\n".join(_result["failures"])
except Exception:
    _result["error"] = traceback.format_exc(limit=2)
finally:
    sys.stdout = _old
_result["stdout"] = _buf.getvalue()[-5000:]
json.dumps(_result)
`
}
