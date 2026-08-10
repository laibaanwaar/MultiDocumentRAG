import axios from 'axios';
import apiClient, { buildAuthUrl } from '../api/apiClient';

export const AUTH_EMAIL_STORAGE_KEY = 'paklaw_auth_email';
export const AUTH_RESEND_COOLDOWN_SECONDS = 60;
export const AUTH_ACCESS_STORAGE_KEY = 'paklaw_access';
export const AUTH_REFRESH_STORAGE_KEY = 'paklaw_refresh';
export const AUTH_USER_STORAGE_KEY = 'paklaw_user';

export async function signupUser(payload) {
  const response = await apiClient.post(buildAuthUrl('signup/'), payload);
  return response.data;
}

export async function verifyOtp(payload) {
  const response = await apiClient.post(buildAuthUrl('verify-email/'), payload);
  return response.data;
}

export async function resendOtp(payload) {
  const response = await apiClient.post(buildAuthUrl('resend-otp/'), payload);
  return response.data;
}

export async function loginUser(payload) {
  const response = await apiClient.post(buildAuthUrl('login/'), payload);
  return response.data;
}

export async function getProfile() {
  const response = await apiClient.get(buildAuthUrl('me/'));
  return response.data;
}

export async function logoutUser(payload) {
  const response = await apiClient.post(buildAuthUrl('logout/'), payload);
  return response.data;
}

export function splitFullName(fullName) {
  const trimmedName = fullName.trim();

  if (!trimmedName) {
    return {
      first_name: '',
      last_name: ''
    };
  }

  const parts = trimmedName.split(/\s+/);

  return {
    first_name: parts[0] || '',
    last_name: parts.slice(1).join(' ')
  };
}

export function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function normalizeMessages(value) {
  if (Array.isArray(value)) {
    return value.map(String);
  }

  if (value == null) {
    return [];
  }

  return [String(value)];
}

export function parseAuthApiError(error, fieldMap = {}, options = {}) {
  const throttleMessage = options.throttleMessage || 'Too many requests. Please wait a moment and try again.';
  const fieldErrors = Object.fromEntries(
    Object.values(fieldMap).map((field) => [field, ''])
  );

  if (!axios.isAxiosError(error)) {
    return {
      fieldErrors,
      message: 'Could not connect to the server. Make sure the Django backend is running.',
      code: '',
      retryAfterSeconds: 0,
      status: 0
    };
  }

  if (!error.response) {
    return {
      fieldErrors,
      message: 'Could not connect to the server. Make sure the Django backend is running.',
      code: '',
      retryAfterSeconds: 0,
      status: 0
    };
  }

  const { status, data } = error.response;
  let message = data?.message || 'Something went wrong. Please try again.';

  if (status === 400 && data?.errors && typeof data.errors === 'object') {
    Object.entries(data.errors).forEach(([backendField, backendMessages]) => {
      const frontendField = fieldMap[backendField];
      const normalized = normalizeMessages(backendMessages).join(' ');

      if (frontendField) {
        fieldErrors[frontendField] = normalized;
      }
    });
  }

  if (status === 429) {
    message = data?.message || throttleMessage;
  }

  if (status >= 500) {
    message = 'Something went wrong on the server. Please try again.';
  }

  return {
    fieldErrors,
    message,
    code: data?.code || '',
    retryAfterSeconds: Number(data?.retry_after_seconds || error.response.headers?.['retry-after'] || 0),
    status
  };
}

export function storeAuthEmail(email) {
  sessionStorage.setItem(AUTH_EMAIL_STORAGE_KEY, email);
}

export function getStoredAuthEmail() {
  return sessionStorage.getItem(AUTH_EMAIL_STORAGE_KEY) || '';
}

export function clearStoredAuthEmail() {
  sessionStorage.removeItem(AUTH_EMAIL_STORAGE_KEY);
}

export function storeAuthSession(data) {
  sessionStorage.setItem(AUTH_ACCESS_STORAGE_KEY, data?.access || '');
  sessionStorage.setItem(AUTH_REFRESH_STORAGE_KEY, data?.refresh || '');
  sessionStorage.setItem(AUTH_USER_STORAGE_KEY, JSON.stringify(data?.user || {}));
}

export function getStoredAccessToken() {
  return sessionStorage.getItem(AUTH_ACCESS_STORAGE_KEY) || '';
}

export function getStoredRefreshToken() {
  return sessionStorage.getItem(AUTH_REFRESH_STORAGE_KEY) || '';
}

export function getStoredUser() {
  const rawUser = sessionStorage.getItem(AUTH_USER_STORAGE_KEY);

  if (!rawUser) {
    return null;
  }

  try {
    return JSON.parse(rawUser);
  } catch {
    return null;
  }
}

export function clearAuthSession() {
  sessionStorage.removeItem(AUTH_ACCESS_STORAGE_KEY);
  sessionStorage.removeItem(AUTH_REFRESH_STORAGE_KEY);
  sessionStorage.removeItem(AUTH_USER_STORAGE_KEY);
  clearStoredAuthEmail();
}

export function isAuthFailure(errorDetails) {
  if (!errorDetails) {
    return false;
  }

  return (
    errorDetails.status === 401 ||
    errorDetails.code === 'TOKEN_INVALID' ||
    errorDetails.code === 'TOKEN_EXPIRED' ||
    errorDetails.code === 'AUTHENTICATION_REQUIRED'
  );
}
