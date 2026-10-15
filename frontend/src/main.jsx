import React, { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import { Activity, BookOpen, Github, Moon, Settings, Sun, UploadCloud } from "lucide-react";
import "react-loading-skeleton/dist/skeleton.css";
import "./styles.css";

const Home = lazy(() => import("./pages/Home.jsx"));
const About = lazy(() => import("./pages/About.jsx"));
const Process = lazy(() => import("./pages/Process.jsx"));
const Results = lazy(() => import("./pages/Results.jsx"));
const EvaluationPage = lazy(() => import("./pages/Evaluation.jsx"));
const SettingsPage = lazy(() => import("./pages/Settings.jsx"));

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  const navItems = useMemo(
    () => [
      { to: "/", label: "Home" },
      { to: "/about", label: "About" },
      { to: "/process", label: "Process" },
      { to: "/results", label: "Results" },
      { to: "/evaluation", label: "Evaluation" },
      { to: "/settings", label: "Settings" },
    ],
    [],
  );

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-zinc-100 text-zinc-950 transition-colors dark:bg-[#07090f] dark:text-zinc-50">
        <div className="fixed inset-0 -z-10 overflow-hidden">
          <div className="absolute left-1/2 top-[-18rem] h-[40rem] w-[64rem] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(20,184,166,.24),transparent_65%)] blur-2xl" />
          <div className="audio-grid absolute inset-0 opacity-50 dark:opacity-30" />
        </div>
        <header className="sticky top-0 z-50 border-b border-black/10 bg-white/70 backdrop-blur-2xl dark:border-white/10 dark:bg-[#07090f]/72">
          <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
            <NavLink to="/" className="flex items-center gap-2 font-semibold">
              <span className="grid h-9 w-9 place-items-center rounded-lg bg-teal-500 text-zinc-950">
                <Activity size={20} />
              </span>
              <span>ArnobSpeechAI</span>
            </NavLink>
            <nav className="hidden items-center gap-1 md:flex">
              {navItems.map((item) => (
                <NavLink key={item.to} to={item.to} className={({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`}>
                  {item.label}
                </NavLink>
              ))}
            </nav>
            <div className="flex items-center gap-2">
              <a className="icon-btn" href="https://github.com/TuhassinulArnob" target="_blank" rel="noreferrer" aria-label="GitHub Repository">
                <Github size={18} />
              </a>
              <a className="icon-btn" href="/about" aria-label="Documentation">
                <BookOpen size={18} />
              </a>
              <button className="icon-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle theme">
                {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
              </button>
            </div>
          </div>
          <nav className="flex gap-1 overflow-x-auto px-4 pb-3 md:hidden">
            {navItems.map((item) => (
              <NavLink key={item.to} to={item.to} className={({ isActive }) => `nav-link whitespace-nowrap ${isActive ? "nav-link-active" : ""}`}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </header>
        <main>
          <Suspense fallback={<div className="mx-auto max-w-7xl px-4 py-20"><div className="skeleton-block h-64" /></div>}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/about" element={<About />} />
              <Route path="/process" element={<Process />} />
              <Route path="/results" element={<Results />} />
              <Route path="/evaluation" element={<EvaluationPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </Suspense>
        </main>
        <Toaster position="top-right" toastOptions={{ className: "toast-surface" }} />
        <a className="fixed bottom-5 right-5 grid h-12 w-12 place-items-center rounded-full bg-teal-400 text-zinc-950 shadow-glow md:hidden" href="/process" aria-label="Upload Audio">
          <UploadCloud size={21} />
        </a>
      </div>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")).render(<App />);
