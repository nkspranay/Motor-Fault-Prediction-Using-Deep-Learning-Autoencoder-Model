const API_BASE = "http://localhost:8000";

export async function fetchData() {
  try {
    const res = await fetch(`${API_BASE}/data`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
  console.log("API DATA:", data);  
  return data;
  } catch (e) {
    return null;
  }
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return await res.json();
  } catch (e) {
    return null;
  }
}