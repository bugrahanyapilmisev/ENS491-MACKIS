import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Card } from "./ui/card";
import { Alert } from "./ui/alert";
import { Mail, Lock, LogIn, UserPlus } from "lucide-react";
import sabancıLogo from "../assets/sabanci_logo.png";
import React from 'react';

// 1. Import our new "Smart Courier" (Axios Instance)
import api from "../api/axios";

interface LoginPageProps {
  onLogin: (email: string, name: string, isAdmin: boolean) => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // 2. Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Basic Validations
    if (!email || !password) {
      setError("Please fill in all fields");
      return;
    }

    if (isSignUp && !name) {
      setError("Please enter your name");
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError("Please enter a valid email address");
      return;
    }

    setIsLoading(true);

    try {
      if (isSignUp) {
        // We haven't implemented the /register endpoint in the backend yet.
        // Blocking registration for now.
        throw new Error("Registration is currently invite-only. Please use the provided admin account.");
      }

      // 3. ACTUAL BACKEND REQUEST (Using Axios)
      // We don't need to manually add headers or full URL; 'api' handles it.
      const response = await api.post("/auth/login", {
        email: email,
        password: password
      });

      // Axios wraps the actual backend response inside the .data property
      const data = response.data;

      // 4. Save the Token
      // IMPORTANT: We use 'access_token' as the key name because our 
      // api/axios.js interceptor looks specifically for localStorage.getItem('access_token')
      localStorage.setItem("access_token", data.access_token);

      // Pass user details to the parent component
      // We handle the case where 'is_admin' might be missing or named differently
      onLogin(
        email,
        data.user_name,
        data.is_admin || data.role === "admin" || false
      );

    } catch (err: any) {
      console.error("Login Error:", err);
      
      // 5. Improved Error Handling for Axios
      // Axios stores the server response error details in err.response
      if (err.response) {
        // Backend usually sends error details in { detail: "..." }
        setError(err.response.data.detail || "Invalid credentials.");
      } else if (err.request) {
        // The request was made but no response was received (Network Error)
        setError("Cannot connect to server. Is the backend running?");
      } else {
        // Something happened in setting up the request that triggered an Error
        setError(err.message || "An unexpected error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-blue-950 flex items-center justify-center p-4">
      <div className="w-full max-w-5xl grid md:grid-cols-2 gap-8 items-center">
        {/* Left side - Branding */}
        <div className="hidden md:block space-y-8">
          <div className="space-y-4">
            <img 
              src={sabancıLogo} 
              alt="Sabancı Universitesi" 
              className="h-24 w-auto"
            />
            <div>
              <h1 className="text-3xl mb-2">AI Chat Assistant</h1>
              <p className="text-muted-foreground leading-relaxed">
                Welcome to Sabancı University's intelligent chat assistant powered by advanced 
                RAG (Retrieval-Augmented Generation) technology.
              </p>
            </div>
          </div>
          
          <div className="space-y-4">
            <p className="text-muted-foreground leading-relaxed">
              Our AI assistant provides accurate, source-backed answers by searching through 
              official university documents, handbooks, and databases. Every response 
              includes citations to the original sources, ensuring transparency and reliability.
            </p>
            
            <p className="text-muted-foreground leading-relaxed">
              Get instant help with admissions, academics, financial aid, campus life, 
              facilities, and more. Your conversations are securely stored and accessible 
              only to you across all your devices.
            </p>
          </div>
        </div>

        {/* Right side - Login Form */}
        <Card className="p-8 shadow-xl">
          <div className="mb-6 text-center md:hidden">
            <img 
              src={sabancıLogo} 
              alt="Sabancı Universitesi" 
              className="h-16 w-auto mx-auto mb-3"
            />
            <h2 className="mb-1">AI Chat Assistant</h2>
            <p className="text-sm text-muted-foreground">Powered by RAG Technology</p>
          </div>

          <div className="mb-6">
            <h2 className="mb-2">{isSignUp ? "Create Account" : "Welcome Back"}</h2>
            <p className="text-sm text-muted-foreground">
              {isSignUp
                ? "Sign up to access the university AI assistant"
                : "Sign in to continue your conversations"}
            </p>
          </div>

          {error && (
            <Alert className="mb-4 border-destructive/50 text-destructive bg-destructive/10">
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignUp && (
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <div className="relative">
                  <UserPlus className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="name"
                    type="text"
                    placeholder="John Doe"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="pl-10"
                    disabled={isLoading}
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="student@sabanciuniv.edu"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="pl-10"
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10"
                  disabled={isLoading}
                />
              </div>
            </div>

            <Button 
              type="submit" 
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
              disabled={isLoading}
            >
              {isLoading ? (
                "Processing..."
              ) : isSignUp ? (
                <>
                  <UserPlus className="h-4 w-4 mr-2" />
                  Create Account
                </>
              ) : (
                <>
                  <LogIn className="h-4 w-4 mr-2" />
                  Sign In
                </>
              )}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={() => {
                setIsSignUp(!isSignUp);
                setError("");
              }}
              className="text-sm text-muted-foreground hover:text-primary transition-colors"
              disabled={isLoading}
            >
              {isSignUp ? (
                <>Already have an account? <span className="text-primary">Sign in</span></>
              ) : (
                <>Don't have an account? <span className="text-primary">Sign up</span></>
              )}
            </button>
          </div>

          <div className="mt-6 pt-6 border-t text-center space-y-2">
            <p className="text-xs text-muted-foreground">
              By continuing, you agree to Sabancı University's terms of service
              and privacy policy
            </p>
            <p className="text-xs text-blue-600 dark:text-blue-400">
              💡 Tip: Use your created account (e.g. admin@sabanci.edu)
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}