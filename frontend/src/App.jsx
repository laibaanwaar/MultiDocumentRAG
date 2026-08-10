import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Chatbot from './pages/Chatbot';
import Home from './pages/Home';
import HowItWorksPage from './pages/HowItWorksPage';
import Login from './pages/Login';
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
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
