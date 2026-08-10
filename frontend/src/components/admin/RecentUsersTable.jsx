import { Link } from 'react-router-dom';

function formatDate(value) {
  if (!value) {
    return '—';
  }

  try {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(new Date(value));
  } catch {
    return '—';
  }
}

function userDisplayName(user) {
  const fullName = [user?.first_name, user?.last_name].filter(Boolean).join(' ').trim();
  return fullName || user?.username || '—';
}

function LoadingRows() {
  return Array.from({ length: 4 }).map((_, index) => (
    <tr key={index} className="border-t border-[#f1eadf]">
      {Array.from({ length: 6 }).map((__, cellIndex) => (
        <td key={cellIndex} className="px-4 py-4">
          <div className="h-4 animate-pulse rounded bg-slate-200" />
        </td>
      ))}
    </tr>
  ));
}

export default function RecentUsersTable({ users = [], loading = false }) {
  return (
    <section className="overflow-hidden rounded-[12px] border border-[#eadfce] bg-white shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
      <div className="flex items-center justify-between border-b border-[#f1eadf] px-5 py-4">
        <h2 className="font-serif text-[20px] font-bold" style={{ color: '#111827' }}>
          Recent Users
        </h2>

        <Link
          to="/admin/users"
          className="rounded-[9px] border border-[#d8c5aa] px-3.5 py-2 text-[12px] font-medium transition hover:bg-[#faf7f1]"
          style={{ color: '#334155' }}
        >
          View All Users
        </Link>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full table-fixed text-left">
          <thead>
            <tr className="text-[11px]" style={{ color: '#64748b' }}>
              <th className="w-[20%] px-4 py-3 font-medium">Username/Name</th>
              <th className="w-[24%] px-4 py-3 font-medium">Email</th>
              <th className="w-[12%] px-4 py-3 font-medium">Status</th>
              <th className="w-[18%] px-4 py-3 font-medium">Subscription Plan</th>
              <th className="w-[14%] px-4 py-3 font-medium">Date Joined</th>
              <th className="w-[12%] px-4 py-3 font-medium">View Details</th>
            </tr>
          </thead>

          <tbody className="text-[12px]" style={{ color: '#334155' }}>
            {loading ? <LoadingRows /> : null}

            {!loading && users.length === 0 ? (
              <tr className="border-t border-[#f1eadf]">
                <td colSpan={6} className="px-4 py-8 text-center text-[14px]" style={{ color: '#64748b' }}>
                  No data available
                </td>
              </tr>
            ) : null}

            {!loading
              ? users.map((user) => (
                  <tr key={user.id} className="border-t border-[#f1eadf] align-middle">
                    <td className="px-4 py-4">
                      <div className="truncate text-[12px] font-semibold" style={{ color: '#111827' }}>
                        {userDisplayName(user)}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <div className="truncate text-[12px]" style={{ color: '#475569' }}>
                        {user.email || '—'}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <span className={`inline-flex rounded-full px-2.5 py-1 text-[11px] ${user.is_active ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                        {user.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>

                    <td className="px-4 py-4">
                      <div className="truncate text-[12px]" style={{ color: '#334155' }}>
                        {user.subscription?.plan_name || '—'}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <div className="text-[12px]" style={{ color: '#475569' }}>
                        {formatDate(user.date_joined)}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <Link to={`/admin/users/${user.id}`} className="text-[12px] font-medium hover:underline" style={{ color: '#9d7438' }}>
                        View Details
                      </Link>
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
