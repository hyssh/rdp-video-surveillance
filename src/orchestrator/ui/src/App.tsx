import { useEffect, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { Empty, ErrorBoundary } from './components/common';
import Dashboard from './pages/Dashboard';
import Insights from './pages/Insights';
import Session from './pages/Session';

function ThemeToggle() {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme ?? 'dark');
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('theme', theme); } catch { /* ignore */ }
  }, [theme]);
  return (
    <button className="icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} title="Toggle theme">
      {theme === 'dark' ? '☀️' : '🌙'}
    </button>
  );
}

export default function App() {
  return (
    <>
      <header className="topbar">
        <NavLink to="/" className="brand"><span className="logo" aria-hidden="true">◉</span><span className="name">RDP Watch</span></NavLink>
        <nav className="nav" aria-label="Main">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/insights">Insights</NavLink>
        </nav>
        <ThemeToggle />
      </header>
      <main>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/session/:id" element={<Session />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="*" element={<div className="page"><Empty icon="🧭" title="Page not found" /></div>} />
          </Routes>
        </ErrorBoundary>
      </main>
    </>
  );
}
