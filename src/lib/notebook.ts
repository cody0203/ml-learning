export const liteUrl = (path: string) => `${import.meta.env.BASE_URL}lite/lab/index.html?path=${encodeURIComponent(path)}`
export const notebookRoute = (path: string) => `/notebook?path=${encodeURIComponent(path)}`
