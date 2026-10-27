"use client";

import React, { useEffect, useRef, useState } from "react";
import { Coffee, ExternalLink, Loader2, Moon, Sun, X } from "lucide-react";

interface KofiModalProps {
  isOpen: boolean;
  onClose: () => void;
  username?: string;
}

export default function KofiModal({
  isOpen,
  onClose,
  username = "ongyiumark",
}: KofiModalProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const triggerElementRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      triggerElementRef.current = document.activeElement as HTMLElement | null;
      setIsLoading(true);
    }
    if (!isOpen) return;

    const modal = modalRef.current;
    if (modal) {
      const focusables = modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (focusables.length > 0) {
        focusables[0].focus();
      }
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab" && modal) {
        const focusables = modal.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      if (triggerElementRef.current) {
        triggerElementRef.current.focus();
      }
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Clean iframe embed without preview wrappers or awkward letterboxing
  const kofiEmbedUrl = `https://ko-fi.com/${username}/?hidefeed=true&widget=true&embed=true`;
  const kofiDirectUrl = `https://ko-fi.com/${username}`;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="kofi-modal-title"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto p-2.5 sm:p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150"
    >
      <div
        ref={modalRef}
        onClick={(e) => e.stopPropagation()}
        className="relative flex w-full max-w-[342px] flex-col overflow-hidden rounded-2xl border border-ocean-600/70 bg-[#0c1a29] shadow-2xl shadow-black/80 my-auto max-h-[calc(100dvh-20px)] sm:max-h-[calc(100dvh-36px)]"
      >
        {/* Header Bar: Perfectly aligned with site's ocean gradient & typography */}
        <div className="shrink-0 flex items-center justify-between border-b border-ocean-700/60 bg-gradient-to-r from-[#173857] via-[#286398] to-[#193d60] px-3.5 py-2.5 shadow-sm">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gold-200/20 border border-gold-300/40 text-gold-200 shadow-sm">
              <Coffee className="h-3.5 w-3.5 text-[#FFE79C]" />
            </div>
            <div>
              <h2
                id="kofi-modal-title"
                className="text-xs font-bold text-white tracking-tight drop-shadow-[1px_1px_2px_#FF8A55]"
              >
                Support on Ko-fi
              </h2>
              <p className="text-[10px] font-medium text-[#FFE79C]/90">
                @{username}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {/* Theme Toggle (Dark to match studio, Light for original) */}
            <button
              type="button"
              onClick={() => setIsDarkMode((prev) => !prev)}
              title={
                isDarkMode
                  ? "Switch to original light widget"
                  : "Switch to dark studio theme"
              }
              aria-label="Toggle widget theme"
              className="flex items-center gap-1 rounded-lg border border-ocean-600/70 bg-[#0e2135]/80 px-2 py-1 text-[11px] font-medium text-sky-200 transition hover:bg-[#163554] hover:text-white"
            >
              {isDarkMode ? (
                <>
                  <Moon className="h-3 w-3 text-gold-200" />
                  <span className="text-[10px]">Dark</span>
                </>
              ) : (
                <>
                  <Sun className="h-3 w-3 text-amber-300" />
                  <span className="text-[10px]">Light</span>
                </>
              )}
            </button>

            {/* Direct External Link */}
            <a
              href={kofiDirectUrl}
              target="_blank"
              rel="noopener noreferrer"
              title="Open full page on ko-fi.com in a new tab"
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-ocean-600/70 bg-[#0e2135]/80 text-sky-200 transition hover:bg-[#163554] hover:text-white"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>

            {/* Close Modal */}
            <button
              onClick={onClose}
              title="Close"
              aria-label="Close modal"
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-ocean-600/70 bg-[#0e2135]/80 text-slate-300 transition hover:bg-rose-950/70 hover:text-rose-200 hover:border-rose-600/60"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Embedded Ko-fi Widget Container: Sized precisely to 342px, scrollable on shorter tablet viewports */}
        <div
          className={`relative flex-1 overflow-y-auto overscroll-contain transition-colors duration-200 ${
            isDarkMode ? "bg-[#0c1a29]" : "bg-[#f5f7f9]"
          }`}
          style={{
            maxHeight: "calc(100dvh - 88px)",
          }}
        >
          {isLoading && (
            <div
              className={`absolute inset-0 z-10 flex flex-col items-center justify-center gap-2.5 ${
                isDarkMode ? "bg-[#0c1a29]" : "bg-[#f5f7f9]"
              }`}
            >
              <Loader2 className="h-7 w-7 animate-spin text-gold-300" />
              <span className="text-xs font-medium text-sky-200/90">
                Loading Ko-fi widget…
              </span>
            </div>
          )}

          <iframe
            id="kofiframe"
            src={kofiEmbedUrl}
            onLoad={() => setIsLoading(false)}
            title="Support on Ko-fi"
            allow="payment *; payment"
            className="w-full border-none transition-all duration-300"
            style={{
              height: "640px",
              minHeight: "640px",
              marginTop: "-8px", // Clean flush alignment: trims top margin so the blue polygon banner sits flush with the header
              filter: isDarkMode
                ? "invert(0.9) hue-rotate(185deg) contrast(1.08) brightness(0.96)"
                : "none",
            }}
          />
        </div>

        {/* Tablet / Mobile Fallback: Direct link if browser security or embedded payment hangs */}
        <div className="shrink-0 border-t border-ocean-700/60 bg-[#081522] px-3 py-1.5 text-center">
          <a
            href={kofiDirectUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-[11px] font-medium text-sky-200/80 hover:text-gold-200 transition"
          >
            <span>Payment slow or freezing?</span>
            <span className="text-[#FFE79C] underline underline-offset-2 flex items-center gap-0.5">
              Open directly on Ko-fi
              <ExternalLink className="h-2.5 w-2.5 inline" />
            </span>
          </a>
        </div>
      </div>
    </div>
  );
}
