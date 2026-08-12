import { registerPyodideBridge } from "@ameva/forge";

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
  
  log("Fetching wheel list...");
  const wheelsRes = await fetch("/api/wheels");
  const wheels = await wheelsRes.json();
  if (wheels.length === 0) throw new Error("No wheel found");
  
  log(`Installing ${wheels[0]}...`);
  await micropip.install("/" + wheels[0]);
  
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
