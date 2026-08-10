import axios from 'axios';

const AUTH_ACCESS_STORAGE_KEY = 'paklaw_access';
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1/auth/';

function normalizeBaseUrl(url) {
  return url.endsWith('/') ? url : `${url}/`;
}

function removeTrailingSegment(url, segment) {
  const normalizedUrl = normalizeBaseUrl(url);
  const trailingSegment = `${segment}/`;

  if (normalizedUrl.endsWith(trailingSegment)) {
    return normalizedUrl.slice(0, -trailingSegment.length);
  }

  return normalizedUrl;
}

export const authBaseUrl = normalizeBaseUrl(configuredBaseUrl);
export const apiRootUrl = normalizeBaseUrl(removeTrailingSegment(authBaseUrl, 'auth'));

export function buildAuthUrl(path = '') {
  return `${authBaseUrl}${path}`;
}

export function buildApiUrl(path = '') {
  return `${apiRootUrl}${path}`;
}

const apiClient = axios.create({
  baseURL: authBaseUrl,
  headers: {
    'Content-Type': 'application/json'
  }
});

apiClient.interceptors.request.use((config) => {
  const accessToken = sessionStorage.getItem(AUTH_ACCESS_STORAGE_KEY) || '';

  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }

  return config;
});

export default apiClient;
