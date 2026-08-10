import { useNavigate } from 'react-router-dom';

const navItems = [
  { label: 'Home', path: '/' },
  { label: 'Features', path: '/#features' },
  { label: 'How It Works', path: '/how-it-works' },
  { label: 'Pricing', path: '/pricing' }
];

export default function SiteHeader({ activeItem = 'Home' }) {
  const navigate = useNavigate();

  return (
    <header className="sticky top-0 z-30 border-b border-white/5 bg-[#07111a]/95 backdrop-blur-xl">
      <div className="mx-auto flex h-18 max-w-[1440px] items-center justify-between px-4 sm:px-6 lg:px-10">
        <div className="leading-tight">
          <div className="text-[18px] font-semibold text-slate-100">PakLaw AI</div>
          <div className="text-[12px] text-slate-400">Multi Doc Chatbot</div>
        </div>

        <nav className="hidden items-center gap-6 xl:flex 2xl:gap-8">
          {navItems.map((item) => {
            const isActive = item.label === activeItem;

            return (
              <button
                key={item.label}
                type="button"
                onClick={() => navigate(item.path)}
                className={[
                  'whitespace-nowrap text-[14px] transition hover:text-amber-200',
                  isActive ? 'text-amber-200' : 'text-slate-300'
                ].join(' ')}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => navigate('/login')}
            className="rounded-xl border border-white/10 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-amber-300/30 hover:bg-white/5"
          >
            Login
          </button>
          <button
            type="button"
            onClick={() => navigate('/signup')}
            className="rounded-xl bg-gradient-to-r from-[#f6d59d] to-[#e0b160] px-4 py-2 text-sm font-semibold text-slate-950 shadow-[0_14px_30px_rgba(224,177,96,0.22)] transition hover:brightness-105"
          >
            Get Started
          </button>
        </div>
      </div>
    </header>
  );
}
