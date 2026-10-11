import { Navigate, Route, Routes } from "react-router-dom";

import { Header } from "./components/Header";
import { QueuePage } from "./pages/QueuePage";
import { ReviewPage } from "./pages/review/ReviewPage";

export function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <Routes>
        <Route path="/" element={<QueuePage />} />
        <Route path="/calls/:callId" element={<ReviewPage />} />
        {/* New call, Memory and Results: next, after the Queue/Review check. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
