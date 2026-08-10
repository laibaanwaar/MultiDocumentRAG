import { useNavigate } from 'react-router-dom';
import signupBg from '../../../assets/legal_scales_building.jpg';
import AuthBranding from './AuthBranding';

const navItems = ['Home', 'Features', 'How It Works', 'Pricing'];

function AuthHeader({ showBack = true }) {
  const navigate = useNavigate();

  return (
    <header className="sticky top-0 z-30 border-b border-white/5 bg-[#07111a]/95 backdrop-blur-xl">
      <div className="mx-auto flex h-18 max-w-[1440px] items-center justify-between px-4 sm:px-6 lg:px-10">
        <div className="leading-tight">
          <div className="text-[18px] font-semibold text-slate-100">PakLaw AI</div>
          <div className="text-[12px] text-slate-400">Multi Doc Chatbot</div>
        </div>

        <nav className="hidden items-center gap-6 xl:flex 2xl:gap-8">
          {navItems.map((item, index) => (
            <button
              key={item}
              type="button"
              onClick={() => navigate(index === 0 ? '/' : '/')}
              className={[
                'whitespace-nowrap text-[14px] text-slate-300 transition hover:text-amber-200',
                index === 0 ? 'text-amber-200' : ''
              ].join(' ')}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          {showBack ? (
            <button
              type="button"
              onClick={() => navigate('/')}
              className="rounded-xl border border-white/10 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-amber-300/30 hover:bg-white/5"
            >
              Back
            </button>
          ) : (
            <button
              type="button"
              onClick={() => navigate('/login')}
              className="rounded-xl border border-white/10 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-amber-300/30 hover:bg-white/5"
            >
              Login
            </button>
          )}

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

export default function AuthLayout({ children, showBack = true, cardWidthClass = 'max-w-[350px]' }) {
  return (
    <>
      <AuthHeader showBack={showBack} />

      <main className="min-h-[calc(100vh-72px)] bg-[#071019] px-3 py-3 sm:px-4 lg:px-6">
        <div className="mx-auto grid min-h-[calc(100vh-96px)] max-w-[820px] overflow-hidden rounded-[20px] bg-[#0a121d] shadow-[0_20px_48px_rgba(0,0,0,0.42)] lg:grid-cols-[0.54fr_0.78fr]">
          <section
            className="relative hidden overflow-hidden lg:flex"
            style={{
              backgroundImage: `linear-gradient(180deg, rgba(5,10,18,0.28) 0%, rgba(5,10,18,0.12) 100%), url(${signupBg})`,
              backgroundPosition: '24% center',
              backgroundRepeat: 'no-repeat',
              backgroundSize: 'cover',
              filter: 'brightness(1.12) saturate(1.06)'
            }}
          >
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(218,171,93,0.12),transparent_18%),linear-gradient(180deg,rgba(3,8,14,0.02),rgba(3,8,14,0.18))]" />
            <div className="relative z-10 flex h-full flex-col px-5 py-6">
              <div>
                <AuthBranding compact />

                <div className="mt-7 max-w-[190px]">
                  <h2 className="font-serif text-[20px] leading-[1.04] text-[#d5a355]">
                    Your Legal Research,
                    <span className="block text-[#f5efe7]">Reimagined.</span>
                  </h2>
                  <p className="mt-2.5 max-w-[178px] text-[10px] leading-4.5 text-slate-200">
                    Join PakLaw AI and simplify your legal research with intelligent,
                    multi-document analysis built for Pakistani lawyers.
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section className="relative flex items-center justify-center bg-[radial-gradient(circle_at_top,rgba(224,177,96,0.04),transparent_16%)] px-3 py-4 sm:px-4 lg:px-5">
            <div className={`signup-card w-full rounded-[18px] bg-[#0d1826]/97 px-3.5 py-3 shadow-[0_16px_38px_rgba(0,0,0,0.32)] backdrop-blur-xl ${cardWidthClass}`}>
              {children}
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
