const API_BASE_URL =
  "https://gene-seen-alot-scsi.trycloudflare.com";

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("Backend health check failed");
  }

  return await response.json();
}