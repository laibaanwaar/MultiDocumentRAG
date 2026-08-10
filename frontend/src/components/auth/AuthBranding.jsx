export function BrandMark({ className = 'h-7 w-7' }) {
  return (
    <svg viewBox="0 0 64 64" fill="none" aria-hidden="true" className={className}>
      <path d="M32 10v26" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M18 18h28" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M16 22 8 35h16L16 22Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      <path d="M48 22 40 35h16L48 22Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      <path d="M24 46h16" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M20 52h24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export default function AuthBranding({ compact = false }) {
  if (compact) {
    return (
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black/18 text-amber-200">
          <BrandMark className="h-6 w-6" />
        </div>
        <div>
          <div className="font-serif text-[20px] text-[#f4e8d5]">PakLaw AI</div>
          <div className="text-[11px] text-slate-300">Multi Doc Chatbot</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black/18 text-amber-200">
        <BrandMark className="h-6 w-6" />
      </div>
      <div>
        <div className="font-serif text-[20px] text-[#f4e8d5]">PakLaw AI</div>
        <div className="text-[11px] text-slate-300">Multi Doc Chatbot</div>
      </div>
    </div>
  );
}
