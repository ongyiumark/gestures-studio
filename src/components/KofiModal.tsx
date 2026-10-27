"use client";

import React, { useEffect, useRef, useState } from "react";
import { Coffee, ExternalLink, Loader2, X } from "lucide-react";

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

  const kofiEmbedUrl = `https://ko-fi.com/${username}/?hidefeed=true&widget=true&embed=true&preview=true`;
  const kofiDirectUrl = `https://ko-fi.com/${username}`;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="kofi-modal-title"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4 backdrop-blur-sm animate-in fade-in duration-150"
    >
      <div
        ref={modalRef}
        onClick={(e) => e.stopPropagation()}
        className="relative flex w-full max-w-[460px] flex-col overflow-hidden rounded-2xl border border-ocean-700/70 bg-[#0c1a29] shadow-2xl shadow-black/60 max-h-[92vh]"
      >
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-ocean-800/80 bg-gradient-to-r from-[#112437] to-[#0c1a29] px-4 py-3 sm:px-5">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[#FF5E5B]/20 border border-[#FF5E5B]/40 text-[#FF5E5B]">
              <Coffee className="h-4 w-4" />
            </div>
            <div>
              <h2
                id="kofi-modal-title"
                className="text-sm font-bold text-white tracking-tight"
              >
                Support Hand Pose Studio
              </h2>
              <p className="text-[11px] text-sky-200/75">
                Support via Ko-fi &bull; @{username}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <a
              href={kofiDirectUrl}
              target="_blank"
              rel="noopener noreferrer"
              title="Open profile on ko-fi.com in a new tab"
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-ocean-700/60 bg-[#0e2135]/80 text-sky-200/80 transition hover:bg-[#163554] hover:text-white"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
            <button
              onClick={onClose}
              title="Close"
              aria-label="Close modal"
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-ocean-700/60 bg-[#0e2135]/80 text-slate-400 transition hover:bg-rose-950/60 hover:text-rose-200 hover:border-rose-700/60"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Embedded Ko-fi Widget Iframe with Loading State */}
        <div className="relative flex-1 min-h-[580px] sm:min-h-[640px] bg-[#0c1a29]">
          {isLoading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2.5 bg-[#0c1a29]">
              <Loader2 className="h-7 w-7 animate-spin text-[#FF5E5B]" />
              <span className="text-xs font-medium text-sky-200/80">
                Loading Ko-fi widget…
              </span>
            </div>
          )}
          <iframe
            id="kofiframe"
            src={kofiEmbedUrl}
            onLoad={() => setIsLoading(false)}
            title="Support on Ko-fi"
            className="w-full h-full min-h-[580px] sm:min-h-[640px] border-none bg-transparent"
            allow="payment"
          />
        </div>
      </div>
    </div>
  );
}
