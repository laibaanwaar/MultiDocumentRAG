import axios from 'axios';
import apiClient, { buildApiUrl } from '../api/apiClient';

const subscriptionRequestCache = new Map();
const subscriptionThrottleCache = new Map();

function normalizeParams(params = {}) {
  const normalized = {};

  Object.entries(params).forEach(([key, value]) => {
    if (value !== '' && value != null) {
      normalized[key] = value;
    }
  });

  return normalized;
}

function buildSubscriptionsCacheKey(params = {}) {
  return JSON.stringify(normalizeParams(params));
}

function createThrottledSubscriptionsError(retryAfterSeconds) {
  return new axios.AxiosError(
    'Too many requests. Please wait a moment and try again.',
    'ERR_BAD_REQUEST',
    undefined,
    undefined,
    {
      status: 429,
      data: {
        message: 'Too many requests. Please wait a moment and try again.',
        retry_after_seconds: retryAfterSeconds
      },
      headers: {
        'retry-after': String(retryAfterSeconds)
      }
    }
  );
}

export async function getAdminDashboard() {
  const response = await apiClient.get(buildApiUrl('admin/dashboard/'));
  return response.data;
}

export async function getAdminUsers(params = {}) {
  const response = await apiClient.get(buildApiUrl('admin/users/'), { params });
  return response.data;
}

export async function getAdminUserDetail(userId) {
  const response = await apiClient.get(buildApiUrl(`admin/users/${userId}/`));
  return response.data;
}

export async function getAdminSubscriptions(params = {}) {
  const response = await apiClient.get(buildApiUrl('admin/subscriptions/'), { params });
  return response.data;
}

export async function getSubscriptions(params = {}) {
  const normalizedParams = normalizeParams(params);
  const cacheKey = buildSubscriptionsCacheKey(normalizedParams);
  const activeThrottleUntil = subscriptionThrottleCache.get(cacheKey) || 0;
  const now = Date.now();

  if (activeThrottleUntil > now) {
    const retryAfterSeconds = Math.max(1, Math.ceil((activeThrottleUntil - now) / 1000));
    throw createThrottledSubscriptionsError(retryAfterSeconds);
  }

  const pendingRequest = subscriptionRequestCache.get(cacheKey);
  if (pendingRequest) {
    return pendingRequest;
  }

  const request = apiClient
    .get(buildApiUrl('admin/subscriptions/'), { params: normalizedParams })
    .then((response) => response.data)
    .catch((error) => {
      if (axios.isAxiosError(error) && error.response?.status === 429) {
        const retryAfterSeconds = Number(error.response?.data?.retry_after_seconds || error.response?.headers?.['retry-after'] || 15);
        subscriptionThrottleCache.set(cacheKey, Date.now() + retryAfterSeconds * 1000);
      }

      throw error;
    })
    .finally(() => {
      subscriptionRequestCache.delete(cacheKey);
    });

  subscriptionRequestCache.set(cacheKey, request);
  return request;
}
