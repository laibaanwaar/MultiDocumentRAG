import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { getStoredAccessToken, getStoredUser } from '../services/authService';

export default function AdminRoute() {
  const location = useLocation();
  const user = getStoredUser();
  const accessToken = getStoredAccessToken();

  if (!accessToken || !user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  if (user.role !== 'admin') {
    return <Navigate to="/chatbot" replace />;
  }

  return <Outlet />;
}
