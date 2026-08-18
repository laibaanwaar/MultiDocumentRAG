import axios from 'axios';
import apiClient, { buildApiUrl } from '../api/apiClient';

const CONNECTION_ERROR_MESSAGE = 'Unable to connect to the server. Please try again.';
const SESSION_EXPIRED_MESSAGE = 'Your session has expired. Please log in again.';
const FORBIDDEN_MESSAGE = 'You do not have permission to perform this action.';
const SERVER_UNAVAILABLE_MESSAGE = 'The server is temporarily unavailable. Please try again.';
const DUPLICATE_PDF_MESSAGE = 'This PDF already exists. Please upload a different file.';
const INVALID_PDF_MESSAGE = 'The selected file is not a valid PDF.';
const FILE_TOO_LARGE_MESSAGE = 'The PDF file must be 50MB or smaller.';
const INVALID_FILE_TYPE_MESSAGE = 'Only PDF files are allowed.';
const INVALID_CATEGORY_MESSAGE = 'The selected category is invalid.';

function normalizeMessages(value) {
  if (Array.isArray(value)) {
    return value.map(String).filter(Boolean);
  }

  if (value == null) {
    return [];
  }

  return [String(value)].filter(Boolean);
}

function humanizeFieldMessage(field, message) {
  const lowerMessage = String(message || '').toLowerCase();

  if (field === 'title') {
    if (lowerMessage.includes('blank') || lowerMessage.includes('required')) {
      return 'Document Title is required.';
    }
  }

  if (field === 'categoryId') {
    if (lowerMessage.includes('inactive')) {
      return 'The selected category is inactive.';
    }

    if (lowerMessage.includes('not found')) {
      return 'The selected category could not be found.';
    }

    if (lowerMessage.includes('required') || lowerMessage.includes('blank')) {
      return 'Category is required.';
    }
  }

  if (field === 'file') {
    if (lowerMessage.includes('only pdf')) {
      return INVALID_FILE_TYPE_MESSAGE;
    }

    if (lowerMessage.includes('not a valid pdf')) {
      return INVALID_PDF_MESSAGE;
    }

    if (lowerMessage.includes('too large') || lowerMessage.includes('maximum allowed size') || lowerMessage.includes('size')) {
      return FILE_TOO_LARGE_MESSAGE;
    }
  }

  return String(message || '').trim();
}

function createEmptyFieldErrors() {
  return {
    title: '',
    categoryId: '',
    file: ''
  };
}

function mapFieldErrors(errors = {}) {
  const fieldErrors = createEmptyFieldErrors();
  const fieldMap = {
    title: 'title',
    category_id: 'categoryId',
    file: 'file'
  };

  Object.entries(errors).forEach(([backendField, backendMessages]) => {
    const frontendField = fieldMap[backendField];

    if (!frontendField) {
      return;
    }

    const firstMessage = normalizeMessages(backendMessages)[0] || '';
    fieldErrors[frontendField] = humanizeFieldMessage(frontendField, firstMessage);
  });

  return fieldErrors;
}

function mapCodeToMessage(code, fallbackMessage) {
  switch (code) {
    case 'AUTHENTICATION_REQUIRED':
    case 'TOKEN_INVALID':
    case 'TOKEN_EXPIRED':
      return SESSION_EXPIRED_MESSAGE;
    case 'FORBIDDEN':
      return FORBIDDEN_MESSAGE;
    case 'DOCUMENT_ALREADY_EXISTS':
      return DUPLICATE_PDF_MESSAGE;
    case 'INVALID_PDF':
      return INVALID_PDF_MESSAGE;
    case 'FILE_TOO_LARGE':
      return FILE_TOO_LARGE_MESSAGE;
    case 'INVALID_FILE_TYPE':
      return INVALID_FILE_TYPE_MESSAGE;
    case 'CATEGORY_NOT_FOUND':
      return INVALID_CATEGORY_MESSAGE;
    case 'CATEGORY_INACTIVE':
      return 'The selected category is inactive.';
    case 'SERVICE_UNAVAILABLE':
      return SERVER_UNAVAILABLE_MESSAGE;
    default:
      return fallbackMessage || '';
  }
}

export function parseDocumentApiError(error, fallbackMessage = 'Something went wrong. Please try again.') {
  const fieldErrors = createEmptyFieldErrors();

  if (!axios.isAxiosError(error)) {
    return {
      code: '',
      status: 0,
      message: CONNECTION_ERROR_MESSAGE,
      fieldErrors
    };
  }

  if (!error.response) {
    return {
      code: '',
      status: 0,
      message: CONNECTION_ERROR_MESSAGE,
      fieldErrors
    };
  }

  const { status, data } = error.response;
  const code = data?.code || '';
  let message = data?.message || fallbackMessage;

  if (status === 401) {
    message = SESSION_EXPIRED_MESSAGE;
  } else if (status === 403) {
    message = FORBIDDEN_MESSAGE;
  } else if (code === 'INVALID_REQUEST' && data?.errors && typeof data.errors === 'object') {
    return {
      code,
      status,
      message: data?.message || 'Please review the highlighted fields and try again.',
      fieldErrors: mapFieldErrors(data.errors)
    };
  } else {
    message = mapCodeToMessage(code, message);
  }

  if (status >= 500) {
    message = SERVER_UNAVAILABLE_MESSAGE;
  }

  return {
    code,
    status,
    message,
    fieldErrors
  };
}

export async function getDocumentCategories() {
  const response = await apiClient.get(buildApiUrl('document-categories/'));
  return response.data;
}

export async function getDocuments(page = 1) {
  const response = await apiClient.get(buildApiUrl('documents/'), {
    params: {
      page
    }
  });

  return response.data;
}

export async function uploadDocument({ title, category_id, file }) {
  const formData = new FormData();
  formData.append('title', title);
  formData.append('category_id', String(category_id));
  formData.append('file', file);

  const response = await apiClient.post(buildApiUrl('documents/'), formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });

  return response.data;
}

export async function deleteDocument(documentId) {
  const response = await apiClient.delete(buildApiUrl(`documents/${documentId}/`));
  return response.data;
}
