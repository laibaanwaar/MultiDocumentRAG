import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Chatbot from './pages/Chatbot';
import Home from './pages/Home';
import HowItWorksPage from './pages/HowItWorksPage';
import Login from './pages/Login';
import AdminLayout from './layouts/AdminLayout';
import AdminRoute from './routes/AdminRoute';
import AdminDashboard from './pages/admin/AdminDashboard';
import AdminDocumentsPage from './pages/admin/AdminDocumentsPage';
import AdminPlans from './pages/admin/AdminPlans';
import AdminSettings from './pages/admin/AdminSettings';
import AdminSubscriptions from './pages/admin/AdminSubscriptions';
import AdminUserDetail from './pages/admin/AdminUserDetail';
import AdminUsers from './pages/admin/AdminUsers';
import PricingPage from './pages/PricingPage';
import ResendOtp from './pages/ResendOtp';
import Signup from './pages/Signup';
import VerifyOtp from './pages/VerifyOtp';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        <Route path="/pricing" element={<PricingPage />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/verify-otp" element={<VerifyOtp />} />
        <Route path="/resend-otp" element={<ResendOtp />} />
        <Route path="/login" element={<Login />} />
        <Route element={<AdminRoute />}>
          <Route path="/admin" element={<AdminLayout />}>
            <Route path="dashboard" element={<AdminDashboard />} />
            <Route path="users" element={<AdminUsers />} />
            <Route path="users/:userId" element={<AdminUserDetail />} />
            <Route path="subscriptions" element={<AdminSubscriptions />} />
            <Route path="plans" element={<AdminPlans />} />
            <Route path="documents" element={<AdminDocumentsPage />} />
            <Route path="settings" element={<AdminSettings />} />
            <Route index element={<Navigate to="/admin/dashboard" replace />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
