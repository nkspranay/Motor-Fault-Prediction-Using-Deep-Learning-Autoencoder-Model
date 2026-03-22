export async function fetchData() {
  try {
    const res = await fetch("http://127.0.0.1:8000/data");
    return await res.json();
  } catch {
    return null;
  }
}