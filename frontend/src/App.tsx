import { Navigate, Route, Routes } from "react-router-dom";

import { Header } from "./components/Header";
import { MemoryPage } from "./pages/memory/MemoryPage";
import { NewCallPage } from "./pages/newcall/NewCallPage";
import { QueuePage } from "./pages/QueuePage";
import { ResultsPage } from "./pages/results/ResultsPage";
import { ReviewPage } from "./pages/review/ReviewPage";

export function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <Routes>
        <Route path="/" element={<QueuePage />} />
        <Route path="/calls/:callId" element={<ReviewPage />} />
        <Route path="/new" element={<NewCallPage />} />
        <Route path="/memory" element={<MemoryPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
