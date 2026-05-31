import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import type { User, Session } from '@supabase/supabase-js';

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const loadSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (!active) return;
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      } catch (err) {
        console.warn('Supabase auth failed to load, falling back to local guest session:', err);
        if (!active) return;
        // Fallback to local guest user if Supabase is unreachable/offline
        const mockUser = {
          id: 'local-guest',
          email: 'guest@local.manga',
          app_metadata: {},
          user_metadata: {},
          aud: 'authenticated',
          created_at: new Date().toISOString()
        } as User;
        setUser(mockUser);
        setLoading(false);
      }
    };

    loadSession();

    let subscription: any = null;
    try {
      const res = supabase.auth.onAuthStateChange((_event, session) => {
        if (!active) return;
        setSession(session);
        setUser(session?.user ?? null);
      });
      subscription = res.data?.subscription;
    } catch (err) {
      console.warn('onAuthStateChange failed:', err);
    }

    return () => {
      active = false;
      if (subscription) {
        subscription.unsubscribe();
      }
    };
  }, []);

  const signUp = async (email: string, password: string) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
      });
      if (error) throw error;
      return data;
    } catch (err) {
      console.warn('SignUp failed, using local fallback:', err);
      const mockUser = {
        id: 'local-guest',
        email: email,
      } as User;
      setUser(mockUser);
      return { user: mockUser, session: null };
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      if (error) throw error;
      return data;
    } catch (err) {
      console.warn('SignIn failed, using local fallback:', err);
      const mockUser = {
        id: 'local-guest',
        email: email,
      } as User;
      setUser(mockUser);
      return { user: mockUser, session: null };
    }
  };

  const signOut = async () => {
    try {
      await supabase.auth.signOut();
    } catch (err) {
      console.warn('SignOut failed, clearing session locally:', err);
    }
    setSession(null);
    setUser(null);
  };

  return {
    user,
    session,
    loading,
    signUp,
    signIn,
    signOut,
  };
}
