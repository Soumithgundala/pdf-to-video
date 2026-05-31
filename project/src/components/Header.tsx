import { useAuth } from '../hooks/useAuth';
import { Film, LogOut } from 'lucide-react';

export function Header() {
  const { user, signOut } = useAuth();

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <Film className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">Manga Recap</span>
          </div>

          <div className="flex items-center gap-4">
            {user && (
              <>
                <div className="text-sm text-slate-400 hidden sm:block">
                  {user.email}
                </div>
                <button
                  onClick={signOut}
                  className="flex items-center gap-2 px-4 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="hidden sm:inline">Sign out</span>
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
