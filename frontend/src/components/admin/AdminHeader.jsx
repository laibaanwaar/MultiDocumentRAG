export default function AdminHeader({ title, subtitle, userName }) {
  return (
    <header className="border-b border-[#182536] bg-[linear-gradient(180deg,#091320_0%,#0b1523_100%)] px-4 py-4 sm:px-5 lg:px-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="font-serif text-[18px] font-bold leading-none text-[#f5efe7] sm:text-[20px] lg:text-[22px]">{title}</h1>
          <p className="mt-1.5 text-[12px] text-slate-400">{subtitle}</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex h-10 items-center gap-3 rounded-[12px] border border-[#233247] bg-[#101b2c] px-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#152132] text-[13px] font-semibold text-amber-200">
              {userName?.[0] || 'A'}
            </div>
            <div className="hidden pr-1 sm:block">
              <div className="text-[12px] font-semibold text-slate-100">{userName}</div>
              <div className="text-[10px] text-slate-400">Administrator</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
