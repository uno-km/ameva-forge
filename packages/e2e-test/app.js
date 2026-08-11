import { registerPyodideBridge } from "@ameva/tensor";

registerPyodideBridge();

async function run() {
  const logDiv = document.getElementById("log");
  const log = (msg) => {
    console.log(msg);
    logDiv.innerHTML += msg + "<br>";
  };
  
  log("Initializing Pyodide...");
  const pyodide = await loadPyodide();
  log("Loading micropip...");
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  
  log("Installing ameva_tensor wheel...");
  await micropip.install("/ameva_tensor-0.1.0-py3-none-any.whl");
  
  log("Fetching Python script...");
  const scriptRes = await fetch("/script.py");
  const scriptCode = await scriptRes.text();
  
  log("Executing Python E2E Test...");
  try {
    // Redirect Python stdout to JS console
    pyodide.setStdout({ batched: (msg) => log("[Python] " + msg) });
    await pyodide.runPythonAsync(scriptCode);
    log("E2E_SUCCESS");
  } catch (err) {
    log("E2E_ERROR: " + err);
  }
}

run();
