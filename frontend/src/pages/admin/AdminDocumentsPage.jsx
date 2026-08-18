import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthMessage from '../../components/auth/AuthMessage';
import DeleteDocumentModal from '../../components/admin/DeleteDocumentModal';
import DocumentToast from '../../components/admin/DocumentToast';
import PdfUploadDropzone from '../../components/admin/PdfUploadDropzone';
import SelectedPdfCard from '../../components/admin/SelectedPdfCard';
import UploadedDocumentsTable from '../../components/admin/UploadedDocumentsTable';
import { clearAuthSession, isAuthFailure } from '../../services/authService';
import {
  deleteDocument,
  getDocumentCategories,
  getDocuments,
  parseDocumentApiError,
  uploadDocument
} from '../../services/documentService';

const INITIAL_FORM_ERRORS = {
  title: '',
  categoryId: '',
  file: ''
};

function hasFormErrors(errors) {
  return Boolean(errors.title || errors.categoryId || errors.file);
}

function createEmptyFormErrors() {
  return {
    title: '',
    categoryId: '',
    file: ''
  };
}

export default function AdminDocumentsPage() {
  const navigate = useNavigate();
  const categoryRequestId = useRef(0);
  const documentRequestId = useRef(0);
  const toastTimerRef = useRef(null);
  const redirectTimerRef = useRef(null);

  const [categories, setCategories] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [count, setCount] = useState(0);
  const [nextPage, setNextPage] = useState(null);
  const [previousPage, setPreviousPage] = useState(null);
  const [page, setPage] = useState(1);
  const [categoriesLoading, setCategoriesLoading] = useState(true);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [title, setTitle] = useState('');
  const [categoryId, setCategoryId] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [formErrors, setFormErrors] = useState(createEmptyFormErrors());
  const [pageError, setPageError] = useState('');
  const [toast, setToast] = useState(null);
  const [documentToDelete, setDocumentToDelete] = useState(null);

  useEffect(
    () => () => {
      if (toastTimerRef.current) {
        window.clearTimeout(toastTimerRef.current);
      }

      if (redirectTimerRef.current) {
        window.clearTimeout(redirectTimerRef.current);
      }
    },
    []
  );

  function showToast(message, variant = 'info') {
    if (!message) {
      return;
    }

    setToast({ message, variant });

    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current);
    }

    toastTimerRef.current = window.setTimeout(() => {
      setToast(null);
    }, 3500);
  }

  function scheduleRedirect(path) {
    if (redirectTimerRef.current) {
      window.clearTimeout(redirectTimerRef.current);
    }

    redirectTimerRef.current = window.setTimeout(() => {
      navigate(path, { replace: true });
    }, 1200);
  }

  function handleSessionExpired(message) {
    clearAuthSession();
    showToast(message, 'error');
    scheduleRedirect('/login');
  }

  function handleForbidden(message) {
    showToast(message, 'error');
    scheduleRedirect('/chatbot');
  }

  function handleApiFailure(error, fallbackMessage) {
    const parsedError = parseDocumentApiError(error, fallbackMessage);

    if (isAuthFailure(parsedError)) {
      handleSessionExpired(parsedError.message);
      return null;
    }

    if (parsedError.status === 403) {
      handleForbidden(parsedError.message);
      return null;
    }

    return parsedError;
  }

  async function loadCategories() {
    const requestId = categoryRequestId.current + 1;
    categoryRequestId.current = requestId;
    setCategoriesLoading(true);

    try {
      const response = await getDocumentCategories();

      if (categoryRequestId.current !== requestId) {
        return;
      }

      setCategories(response?.results || []);
    } catch (error) {
      const parsedError = handleApiFailure(error, 'Unable to load document categories.');

      if (!parsedError || categoryRequestId.current !== requestId) {
        return;
      }

      showToast(parsedError.message, 'error');
      setCategories([]);
    } finally {
      if (categoryRequestId.current === requestId) {
        setCategoriesLoading(false);
      }
    }
  }

  async function loadDocuments(targetPage = 1) {
    const requestId = documentRequestId.current + 1;
    documentRequestId.current = requestId;
    setDocumentsLoading(true);
    setPageError('');

    try {
      const response = await getDocuments(targetPage);

      if (documentRequestId.current !== requestId) {
        return;
      }

      setDocuments(response?.results || []);
      setCount(response?.count || 0);
      setNextPage(response?.next || null);
      setPreviousPage(response?.previous || null);
      setPage(targetPage);
    } catch (error) {
      const parsedError = handleApiFailure(error, 'Unable to load documents.');

      if (!parsedError || documentRequestId.current !== requestId) {
        return;
      }

      setPageError(parsedError.message);
      showToast(parsedError.message, 'error');
    } finally {
      if (documentRequestId.current === requestId) {
        setDocumentsLoading(false);
      }
    }
  }

  useEffect(() => {
    loadCategories();
    loadDocuments(1);
  }, []);

  function handleTitleChange(event) {
    setTitle(event.target.value);
    if (formErrors.title) {
      setFormErrors((currentErrors) => ({ ...currentErrors, title: '' }));
    }
  }

  function handleCategoryChange(event) {
    setCategoryId(event.target.value);
    if (formErrors.categoryId) {
      setFormErrors((currentErrors) => ({ ...currentErrors, categoryId: '' }));
    }
  }

  function handleFileSelect(file) {
    setSelectedFile(file);
    if (file && formErrors.file) {
      setFormErrors((currentErrors) => ({ ...currentErrors, file: '' }));
    }
  }

  function handleFileValidationError(message) {
    setFormErrors((currentErrors) => ({ ...currentErrors, file: message || '' }));
  }

  function validateForm() {
    const nextErrors = createEmptyFormErrors();

    if (!title.trim()) {
      nextErrors.title = 'Document Title is required.';
    }

    if (!categoryId) {
      nextErrors.categoryId = 'Category is required.';
    }

    if (!selectedFile) {
      nextErrors.file = 'Please choose a PDF file.';
    }

    return nextErrors;
  }

  async function handleSubmitDocument() {
    const nextErrors = validateForm();
    setFormErrors(nextErrors);

    if (hasFormErrors(nextErrors) || uploading) {
      showToast('Please complete the form before uploading.', 'error');
      return;
    }

    setUploading(true);

    try {
      await uploadDocument({
        title: title.trim(),
        category_id: categoryId,
        file: selectedFile
      });

      showToast('Document uploaded successfully.', 'success');
      setTitle('');
      setCategoryId('');
      setSelectedFile(null);
      setFormErrors(createEmptyFormErrors());
      await loadDocuments(1);
    } catch (error) {
      const parsedError = handleApiFailure(error, 'Unable to upload the document.');

      if (!parsedError) {
        return;
      }

      const nextFieldErrors = {
        ...createEmptyFormErrors(),
        ...parsedError.fieldErrors
      };

      if (['DOCUMENT_ALREADY_EXISTS', 'INVALID_PDF', 'FILE_TOO_LARGE', 'INVALID_FILE_TYPE'].includes(parsedError.code)) {
        nextFieldErrors.file = parsedError.message;
      }

      if (['CATEGORY_NOT_FOUND', 'CATEGORY_INACTIVE'].includes(parsedError.code)) {
        nextFieldErrors.categoryId = parsedError.message;
      }

      setFormErrors((currentErrors) => ({
        ...currentErrors,
        ...nextFieldErrors
      }));

      showToast(parsedError.message, 'error');
    } finally {
      setUploading(false);
    }
  }

  function handleDeleteClick(document) {
    setDocumentToDelete(document);
  }

  function handleCancelDelete() {
    if (deleting) {
      return;
    }

    setDocumentToDelete(null);
  }

  async function handleConfirmDelete() {
    if (!documentToDelete || deleting) {
      return;
    }

    const targetDocument = documentToDelete;
    const nextValidPage = Math.max(1, Math.min(page, Math.ceil((count - 1) / 5)));

    setDeleting(true);

    try {
      await deleteDocument(targetDocument.id);
      showToast('Document deleted successfully.', 'success');
      setDocumentToDelete(null);
      await loadDocuments(nextValidPage);
    } catch (error) {
      const parsedError = handleApiFailure(error, 'Unable to delete the document.');

      if (!parsedError) {
        return;
      }

      showToast(parsedError.message, 'error');
    } finally {
      setDeleting(false);
    }
  }

  const canSubmitDocument =
    Boolean(title.trim()) &&
    Boolean(categoryId) &&
    Boolean(selectedFile) &&
    !uploading &&
    !categoriesLoading &&
    categories.length > 0;

  return (
    <div className="mx-auto max-w-[1220px] space-y-5">
      <DocumentToast
        message={toast?.message || ''}
        variant={toast?.variant || 'info'}
        onClose={() => setToast(null)}
      />

      {pageError ? <AuthMessage variant="error">{pageError}</AuthMessage> : null}

      <section className="rounded-[12px] border border-[#eadfce] bg-white p-4 shadow-[0_8px_18px_rgba(15,23,42,0.04)] sm:p-5">
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <label className="block text-[12px] font-semibold uppercase tracking-[0.12em]" style={{ color: '#64748b' }}>
              Document Title
            </label>
            <input
              value={title}
              onChange={handleTitleChange}
              placeholder="Enter document title"
              className="mt-2 w-full rounded-[10px] border border-[#e3d8c7] bg-white px-3 py-3 text-[14px] outline-none placeholder:text-[#94a3b8]"
              style={{ color: '#111827' }}
            />
            {formErrors.title ? <p className="mt-2 text-[12px] text-[#9a5f4a]">{formErrors.title}</p> : null}
          </div>

          <div>
            <label className="block text-[12px] font-semibold uppercase tracking-[0.12em]" style={{ color: '#64748b' }}>
              Category
            </label>
            <select
              value={categoryId}
              onChange={handleCategoryChange}
              disabled={categoriesLoading || categories.length === 0}
              className="mt-2 w-full rounded-[10px] border border-[#e3d8c7] bg-white px-3 py-3 text-[14px] outline-none disabled:cursor-not-allowed disabled:bg-[#fbfaf7]"
              style={{ color: '#111827' }}
            >
              <option value="">
                {categoriesLoading ? 'Loading categories...' : categories.length === 0 ? 'No categories available' : 'Select a category'}
              </option>
              {categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
            {formErrors.categoryId ? <p className="mt-2 text-[12px] text-[#9a5f4a]">{formErrors.categoryId}</p> : null}
          </div>
        </div>

        <div className="mt-4">
          <PdfUploadDropzone
            onFileSelect={handleFileSelect}
            onValidationError={handleFileValidationError}
            disabled={uploading}
          />
          {formErrors.file ? <p className="mt-2 text-[12px] text-[#9a5f4a]">{formErrors.file}</p> : null}
        </div>

        {selectedFile ? (
          <div className="mt-4">
            <SelectedPdfCard
              file={selectedFile}
              onSubmit={handleSubmitDocument}
              loading={uploading}
              disabled={!canSubmitDocument}
            />
          </div>
        ) : null}
      </section>

      <UploadedDocumentsTable
        documents={documents}
        loading={documentsLoading}
        count={count}
        page={page}
        nextPage={nextPage}
        previousPage={previousPage}
        onPreviousPage={() => {
          if (previousPage && !documentsLoading && page > 1) {
            loadDocuments(page - 1);
          }
        }}
        onNextPage={() => {
          if (nextPage && !documentsLoading) {
            loadDocuments(page + 1);
          }
        }}
        onDeleteClick={handleDeleteClick}
      />

      <DeleteDocumentModal
        isOpen={Boolean(documentToDelete)}
        documentName={documentToDelete?.original_filename}
        loading={deleting}
        onCancel={handleCancelDelete}
        onDelete={handleConfirmDelete}
      />
    </div>
  );
}
