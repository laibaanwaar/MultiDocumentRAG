import { useRef, useState } from 'react';

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;

function validatePdfFile(file) {
  if (!file) {
    return 'Please choose a PDF file.';
  }

  const isPdfMime = file.type === 'application/pdf';
  const isPdfExtension = file.name?.toLowerCase().endsWith('.pdf');

  if (!isPdfMime && !isPdfExtension) {
    return 'Only PDF files are allowed.';
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return 'Maximum file size is 50MB.';
  }

  return '';
}

export default function PdfUploadDropzone({ onFileSelect, onValidationError, disabled = false }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  function emitFile(file) {
    if (disabled) {
      return;
    }

    const validationError = validatePdfFile(file);

    if (validationError) {
      onFileSelect(null);
      onValidationError(validationError);
      if (inputRef.current) {
        inputRef.current.value = '';
      }
      return;
    }

    onValidationError('');
    onFileSelect(file);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }

  function handleInputChange(event) {
    emitFile(event.target.files?.[0] || null);
  }

  function handleDragOver(event) {
    if (disabled) {
      return;
    }

    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(event) {
    if (disabled) {
      return;
    }

    event.preventDefault();
    setIsDragging(false);
    emitFile(event.dataTransfer.files?.[0] || null);
  }

  function openPicker() {
    if (!disabled) {
      inputRef.current?.click();
    }
  }

  return (
    <section
      className={[
        'rounded-[12px] border bg-[#fbf8f2] p-4 transition sm:p-5',
        disabled ? 'cursor-not-allowed opacity-70' : '',
        isDragging ? 'border-[#9d7438] shadow-[0_10px_22px_rgba(157,116,56,0.12)]' : 'border-[#e7dbc7]'
      ].join(' ')}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input ref={inputRef} type="file" accept="application/pdf,.pdf" className="hidden" onChange={handleInputChange} disabled={disabled} />

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 flex-none items-center justify-center rounded-[12px] bg-[#f3ead8] text-[#9d7438]">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
              <path d="M12 4v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              <path d="m8.5 7.5 3.5-3.5 3.5 3.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M6 14.5v3A1.5 1.5 0 0 0 7.5 19h9A1.5 1.5 0 0 0 18 17.5v-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          <div>
            <h2 className="font-serif text-[19px] font-bold" style={{ color: '#111827' }}>
              PDF File
            </h2>
            <p className="mt-1 text-[12px]" style={{ color: '#64748b' }}>
              Drag and drop a PDF file here or choose one from your device.
            </p>
            <p className="mt-1 text-[11px] uppercase tracking-[0.12em]" style={{ color: '#94a3b8' }}>
              PDF only - Maximum size 50MB
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={openPicker}
          disabled={disabled}
          className="inline-flex h-10 items-center justify-center rounded-[10px] border border-[#d8c5aa] bg-white px-4 text-[13px] font-medium transition hover:bg-[#fcfaf6] disabled:cursor-not-allowed disabled:opacity-60"
          style={{ color: '#334155' }}
        >
          Choose PDF
        </button>
      </div>

      <div
        className={[
          'mt-4 flex min-h-[180px] flex-col items-center justify-center rounded-[12px] border border-dashed px-4 text-center transition',
          disabled ? 'cursor-not-allowed' : '',
          isDragging ? 'border-[#9d7438] bg-[#f9f3e6]' : 'border-[#decfb8] bg-white'
        ].join(' ')}
      >
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-9 w-9 text-[#9d7438]">
          <path d="M12 4v10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path d="m8.5 7.5 3.5-3.5 3.5 3.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M5.5 15.5v1A2.5 2.5 0 0 0 8 19h8a2.5 2.5 0 0 0 2.5-2.5v-1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>

        <p className="mt-3 text-[14px] font-medium" style={{ color: '#111827' }}>
          Drag and drop your PDF here
        </p>
        <p className="mt-1 text-[12px]" style={{ color: '#64748b' }}>
          or use the Choose PDF button above.
        </p>
      </div>
    </section>
  );
}
