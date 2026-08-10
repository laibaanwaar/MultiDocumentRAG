import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthMessage from '../components/auth/AuthMessage';
import ChatHeader from '../components/chatbot/ChatHeader';
import ChatInput from '../components/chatbot/ChatInput';
import ChatSidebar from '../components/chatbot/ChatSidebar';
import ChatWindow from '../components/chatbot/ChatWindow';
import DocumentList from '../components/chatbot/DocumentList';
import {
  clearAuthSession,
  getProfile,
  getStoredRefreshToken,
  getStoredUser,
  isAuthFailure,
  logoutUser,
  parseAuthApiError
} from '../services/authService';
import { askLegalQuestion, fetchChatHistory } from '../services/chatService';

function formatTime(isoDate) {
  if (!isoDate) {
    return '';
  }

  try {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit'
    }).format(new Date(isoDate));
  } catch {
    return '';
  }
}

function getDisplayName(user) {
  if (!user) {
    return 'User';
  }

  const fullName = [user.first_name, user.last_name].filter(Boolean).join(' ').trim();
  return fullName || user.username || 'User';
}

export default function Chatbot() {
  const navigate = useNavigate();
  const [profile, setProfile] = useState(getStoredUser());
  const [historyItems, setHistoryItems] = useState([]);
  const [documents] = useState([]);
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [question, setQuestion] = useState('');
  const [pendingQuestion, setPendingQuestion] = useState('');
  const [pendingAnswer, setPendingAnswer] = useState('');
  const [pendingSources, setPendingSources] = useState([]);
  const [pageError, setPageError] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [questionError, setQuestionError] = useState('');
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);

  const displayName = useMemo(() => getDisplayName(profile), [profile]);

  function handleAuthFailure(message) {
    clearAuthSession();
    navigate('/login', {
      replace: true,
      state: {
        identifier: profile?.email || ''
      }
    });
    return message;
  }

  async function loadChatbotData() {
    setIsBootstrapping(true);
    setPageError('');
    setHistoryError('');

    try {
      const [profileResponse, historyResponse] = await Promise.all([
        getProfile(),
        fetchChatHistory()
      ]);

      setProfile(profileResponse?.data || getStoredUser());
      setHistoryItems(historyResponse?.results || []);
    } catch (error) {
      const parsedError = parseAuthApiError(error);
      if (isAuthFailure(parsedError)) {
        handleAuthFailure(parsedError.message);
        return;
      }

      setPageError(parsedError.message);
    } finally {
      setIsBootstrapping(false);
    }
  }

  useEffect(() => {
    const currentUser = getStoredUser();
    if (!currentUser) {
      navigate('/login', { replace: true });
      return;
    }

    loadChatbotData();
  }, [navigate]);

  async function refreshHistory() {
    setIsLoadingHistory(true);
    setHistoryError('');

    try {
      const historyResponse = await fetchChatHistory();
      setHistoryItems(historyResponse?.results || []);
    } catch (error) {
      const parsedError = parseAuthApiError(error);
      if (isAuthFailure(parsedError)) {
        handleAuthFailure(parsedError.message);
        return;
      }

      setHistoryError(parsedError.message);
    } finally {
      setIsLoadingHistory(false);
    }
  }

  function handleNewChat() {
    setSelectedHistory(null);
    setPendingQuestion('');
    setPendingAnswer('');
    setPendingSources([]);
    setQuestion('');
    setQuestionError('');
  }

  async function handleAskQuestion(event) {
    event.preventDefault();

    if (isAnswering) {
      return;
    }

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setQuestionError('Please enter your legal question.');
      return;
    }

    setQuestionError('');
    setPageError('');
    setSelectedHistory(null);
    setPendingQuestion(trimmedQuestion);
    setPendingAnswer('');
    setPendingSources([]);
    setIsAnswering(true);

    try {
      const response = await askLegalQuestion({
        question: trimmedQuestion
      });

      const responseData = response?.data || {};
      setPendingAnswer(responseData.answer || '');
      setPendingSources(responseData.sources || []);
      setQuestion('');
      await refreshHistory();
    } catch (error) {
      const parsedError = parseAuthApiError(error, {
        question: 'question'
      });

      if (isAuthFailure(parsedError)) {
        handleAuthFailure(parsedError.message);
        return;
      }

      setQuestionError(parsedError.fieldErrors.question || parsedError.message);
      setPendingQuestion('');
      setPendingAnswer('');
      setPendingSources([]);
    } finally {
      setIsAnswering(false);
    }
  }

  async function handleLogout() {
    const refreshToken = getStoredRefreshToken();

    try {
      if (refreshToken) {
        await logoutUser({ refresh: refreshToken });
      }
    } catch {
      // Clear local auth data even if backend logout fails.
    } finally {
      clearAuthSession();
      navigate('/login', { replace: true });
    }
  }

  if (isBootstrapping) {
    return (
      <main className="h-screen overflow-hidden bg-[#06111d] text-slate-200">
        <div className="flex h-full items-center justify-center">
          Loading your chatbot...
        </div>
      </main>
    );
  }

  return (
    <main className="h-screen overflow-hidden bg-[#06111d] text-slate-100">
      <div className="grid h-screen overflow-hidden lg:grid-cols-[218px_1fr]">
        <ChatSidebar
          historyItems={historyItems}
          isLoadingHistory={isLoadingHistory}
          selectedHistoryId={selectedHistory?.id || null}
          onSelectHistory={(historyItem) => {
            setSelectedHistory(historyItem);
            setPendingQuestion('');
            setPendingAnswer('');
            setPendingSources([]);
            setQuestionError('');
          }}
          onRefreshHistory={refreshHistory}
          onNewChat={handleNewChat}
          formatTime={formatTime}
        />

        <section className="flex h-screen min-h-0 flex-col overflow-hidden bg-[linear-gradient(180deg,#081321_0%,#091321_100%)]">
          <ChatHeader
            title={selectedHistory ? 'Chat History' : 'New Chat'}
            userName={displayName}
            onLogout={handleLogout}
          />

          <div className="flex-1 overflow-y-auto px-6 py-5">
            {pageError ? (
              <div className="mb-3">
                <AuthMessage variant="error">{pageError}</AuthMessage>
              </div>
            ) : null}

            <DocumentList documents={documents} />

            <ChatWindow
              selectedHistory={selectedHistory}
              pendingQuestion={pendingQuestion}
              pendingAnswer={pendingAnswer}
              pendingSources={pendingSources}
              isAnswering={isAnswering}
              formatTime={formatTime}
            />

            {historyError ? (
              <div className="mt-3">
                <AuthMessage variant="error">{historyError}</AuthMessage>
              </div>
            ) : null}
          </div>

          <ChatInput
            value={question}
            onChange={(event) => {
              setQuestion(event.target.value);
              setQuestionError('');
            }}
            onSubmit={handleAskQuestion}
            disabled={isAnswering}
            error={questionError}
          />
        </section>
      </div>
    </main>
  );
}
