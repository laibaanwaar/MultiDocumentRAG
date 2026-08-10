import { Link } from 'react-router-dom';

function LoadingRows() {
  return Array.from({ length: 3 }).map((_, index) => (
    <tr key={index} className="border-t border-[#f1eadf]">
      {Array.from({ length: 3 }).map((__, cellIndex) => (
        <td key={cellIndex} className="px-4 py-4">
          <div className="h-4 animate-pulse rounded bg-slate-200" />
        </td>
      ))}
    </tr>
  ));
}

export default function SubscriptionOverview({ plans = [], loading = false }) {
  return (
    <section className="overflow-hidden rounded-[12px] border border-[#eadfce] bg-white shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
      <div className="flex items-center justify-between border-b border-[#f1eadf] px-5 py-4">
        <h2 className="font-serif text-[20px] font-bold" style={{ color: '#111827' }}>
          Subscription Overview
        </h2>

        <Link
          to="/admin/plans"
          className="rounded-[9px] border border-[#d8c5aa] px-3.5 py-2 text-[12px] font-medium transition hover:bg-[#faf7f1]"
          style={{ color: '#334155' }}
        >
          View All Plans
        </Link>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full table-fixed text-left">
          <thead>
            <tr className="text-[11px]" style={{ color: '#64748b' }}>
              <th className="w-[46%] px-4 py-3 font-medium">Plan</th>
              <th className="w-[22%] px-4 py-3 font-medium">Users</th>
              <th className="w-[32%] px-4 py-3 font-medium">Status</th>
            </tr>
          </thead>

          <tbody className="text-[12px]" style={{ color: '#334155' }}>
            {loading ? <LoadingRows /> : null}

            {!loading && plans.length === 0 ? (
              <tr className="border-t border-[#f1eadf]">
                <td colSpan={3} className="px-4 py-8 text-center text-[14px]" style={{ color: '#64748b' }}>
                  No data available
                </td>
              </tr>
            ) : null}

            {!loading
              ? plans.map((plan) => (
                  <tr key={plan.plan_id || plan.code} className="border-t border-[#f1eadf] align-middle">
                    <td className="px-4 py-4">
                      <div className="text-[12px] font-semibold" style={{ color: '#111827' }}>
                        {plan.name}
                      </div>
                      <div className="mt-1 text-[11px] uppercase tracking-[0.12em]" style={{ color: '#94a3b8' }}>
                        {plan.code}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <div className="text-[12px]" style={{ color: '#334155' }}>
                        {plan.subscribers_count ?? '—'}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <span className={`inline-flex rounded-full px-2.5 py-1 text-[11px] ${plan.is_active ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                        {plan.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                  </tr>
                ))
              : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
