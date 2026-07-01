import type { ComfyApp } from "@comfyorg/comfyui-frontend-types";
import React, { Suspense } from "react";
import ReactDOM from "react-dom/client";

import App from "./App";

declare global {
  interface Window {
    app?: ComfyApp;
  }
}

function waitForApp(): Promise<ComfyApp> {
  return new Promise((resolve, reject) => {
    if (window.app) {
      resolve(window.app);
      return;
    }
    const interval = setInterval(() => {
      if (window.app) {
        clearInterval(interval);
        resolve(window.app);
      }
    }, 50);
    setTimeout(() => {
      clearInterval(interval);
      reject(new Error("Timed out waiting for ComfyUI's app to initialize"));
    }, 5000);
  });
}

async function registerRealtimeSidebarTab(): Promise<void> {
  const app = await waitForApp();
  app.extensionManager.registerSidebarTab({
    id: "comfyui-realtime",
    icon: "pi pi-microphone",
    title: "Realtime",
    tooltip: "ComfyUI Realtime",
    type: "custom",
    render: (element: HTMLElement) => {
      const container = document.createElement("div");
      container.style.height = "100%";
      element.appendChild(container);
      ReactDOM.createRoot(container).render(
        <React.StrictMode>
          <Suspense fallback={<div>Loading...</div>}>
            <App />
          </Suspense>
        </React.StrictMode>
      );
    },
  });
}

void registerRealtimeSidebarTab().catch((err) => {
  console.error("ComfyUI-Realtime sidebar tab failed to register:", err);
});
