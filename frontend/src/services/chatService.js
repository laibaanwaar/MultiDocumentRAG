import apiClient, { buildApiUrl } from '../api/apiClient';

export async function fetchChatHistory(page = 1) {
  const response = await apiClient.get(buildApiUrl(`rag/history/?page=${page}`));
  return response.data;
}

export async function askLegalQuestion(payload) {
  const response = await apiClient.post(buildApiUrl('rag/query/'), payload);
  return response.data;
}
