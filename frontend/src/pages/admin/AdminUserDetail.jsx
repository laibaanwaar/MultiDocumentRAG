import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import AuthMessage from '../../components/auth/AuthMessage';
import { getAdminUserDetail } from '../../services/adminService';
import { clearAuthSession, isAuthFailure, parseAuthApiError } from '../../services/authService';

function formatDateTime(value) {
  if (!value) {
    return '—';
  }

  try {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    }).format(new Date(value));
  } catch {
    return '—';
  }
}

function DetailItem({ label, value }) {
  return (
    <div className="rounded-[12px] border border-[#eadfce] bg-white px-4 py-4">
      <div className="text-[12px] text-slate-500">{label}</div>
      <div className="mt-2 text-[14px] font-medium text-slate-800">{value || '—'}</div>
    </div>
  );
}

export default function AdminUserDetail() {
  const { userId } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function loadDetail() {
      setLoading(true);
      setError('');

      try {
        const response = await getAdminUserDetail(userId);
        setDetail(response?.data || null);
      } catch (apiError) {
        const parsedError = parseAuthApiError(apiError);

        if (isAuthFailure(parsedError)) {
          clearAuthSession();
          navigate('/login', { replace: true });
          return;
        }

        if (parsedError.status === 403) {
          navigate('/chatbot', { replace: true });
          return;
        }

        setError(parsedError.status === 0 ? 'Unable to load dashboard data.' : parsedError.message || 'Unable to load dashboard data.');
      } finally {
        setLoading(false);
      }
    }

    loadDetail();
  }, [navigate, userId]);

  const user = detail?.user;
  const subscription = detail?.subscription;

  return (
    <div className="space-y-5">
      {error ? <AuthMessage variant="error">{error}</AuthMessage> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <DetailItem label="Username" value={loading ? 'Loading...' : user?.username || '—'} />
        <DetailItem label="Email" value={loading ? 'Loading...' : user?.email || '—'} />
        <DetailItem label="Joined" value={loading ? 'Loading...' : formatDateTime(user?.date_joined)} />
        <DetailItem label="Status" value={loading ? 'Loading...' : user?.is_active ? 'Active' : 'Inactive'} />
        <DetailItem label="Subscription Plan" value={loading ? 'Loading...' : subscription?.plan_name || '—'} />
        <DetailItem label="Queries Used" value={loading ? 'Loading...' : subscription?.queries_used?.toString() || '—'} />
      </div>

      <section className="rounded-[12px] border border-[#eadfce] bg-white px-5 py-5 shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
        <h2 className="font-serif text-[24px] text-[#111827]">Recent Queries</h2>
        <div className="mt-4 space-y-3">
          {loading ? (
            Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="h-16 animate-pulse rounded-[10px] bg-slate-100" />
            ))
          ) : detail?.rag_usage?.recent_queries?.length ? (
            detail.rag_usage.recent_queries.map((query) => (
              <div key={query.id} className="rounded-[10px] border border-[#f1eadf] px-4 py-4">
                <div className="text-[14px] font-medium text-slate-800">{query.question}</div>
                <div className="mt-2 text-[13px] text-slate-500">{query.answer_preview || '—'}</div>
                <div className="mt-2 text-[11px] text-slate-400">{formatDateTime(query.created_at)}</div>
              </div>
            ))
          ) : (
            <div className="rounded-[10px] border border-dashed border-[#dccdb7] px-4 py-8 text-center text-[14px] text-slate-500">
              No data available
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
