import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import AdminHeader from '../components/admin/AdminHeader';
import AdminSidebar from '../components/admin/AdminSidebar';
import {
  clearAuthSession,
  getStoredRefreshToken,
  getStoredUser,
  logoutUser
} from '../services/authService';

const pageMeta = {
  '/admin/dashboard': {
    title: 'Admin Dashboard',
    subtitle: 'Monitor users, subscriptions, and platform activity.'
  },
  '/admin/users': {
    title: 'Users',
    subtitle: 'Review registered users and their current subscription state.'
  },
  '/admin/subscriptions': {
    title: 'Subscriptions',
    subtitle: 'Inspect active and inactive subscriber records.'
  },
  '/admin/plans': {
    title: 'Plans',
    subtitle: 'Plan management UI is ready for backend list integration.'
  },
  '/admin/documents': {
    title: 'Upload PDF Documents',
    subtitle: 'Upload and manage legal documents for AI search.'
  },
  '/admin/settings': {
    title: 'Settings',
    subtitle: 'Configure future admin preferences and platform settings.'
  }
};

export default function AdminLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const user = getStoredUser();
  const baseMeta = pageMeta[location.pathname] || {
    title: 'Admin',
    subtitle: 'Manage the PakLaw AI platform.'
  };

  async function handleLogout() {
    const refreshToken = getStoredRefreshToken();

    try {
      if (refreshToken) {
        await logoutUser({ refresh: refreshToken });
      }
    } catch {
      // Local auth state should still be cleared.
    } finally {
      clearAuthSession();
      navigate('/login', { replace: true });
    }
  }

  return (
    <main className="min-h-screen bg-[#f5f1ea] lg:grid lg:grid-cols-[204px_minmax(0,1fr)]">
      <AdminSidebar onLogout={handleLogout} />

      <div className="min-w-0 bg-[#f5f1ea]">
        <AdminHeader title={baseMeta.title} subtitle={baseMeta.subtitle} userName={user?.username || 'Admin'} />
        <div className="px-4 py-4 sm:px-5 lg:px-6 lg:py-5">
          <Outlet />
        </div>
      </div>
    </main>
  );
}
