import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, BarChart3, BookOpen, Scale } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-neutral-50">
      {/* Navigation */}
      <header className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="container flex h-14 items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl">
            <span className="bg-primary text-primary-foreground px-2 py-1 rounded">SANR</span>
            <span>Embed</span>
          </div>
          <nav className="flex gap-6 text-sm font-medium">
            <Link href="/benchmark" className="hover:text-primary transition-colors">Benchmark</Link>
            <Link href="/data" className="hover:text-primary transition-colors">Digital Archive</Link>
            <Link href="/embeddings" className="hover:text-primary transition-colors">Embedding Space</Link>
            <Link href="/about" className="hover:text-primary transition-colors">About</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="w-full py-24 lg:py-32 bg-white">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                  Benchmarking Historical Legal Understanding
                </h1>
                <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
                  Evaluating how modern Language Models reason over 17th-century Spanish American legal texts.
                  Measuring accuracy, cross-lingual bias, and drift.
                </p>
              </div>
              <div className="space-x-4">
                <Link href="/benchmark">
                  <Button size="lg" className="gap-2">
                    View Leaderboard <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
                <Link href="/data">
                  <Button variant="outline" size="lg" className="gap-2">
                    Explore Dataset <BookOpen className="w-4 h-4" />
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* Key Features Grid */}
        <section className="w-full py-12 md:py-24 lg:py-32 bg-gray-50">
          <div className="container px-4 md:px-6">
            <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-3">
              <div className="flex flex-col items-center space-y-4 text-center p-6 bg-white rounded-xl shadow-sm border">
                <div className="p-3 bg-blue-50 rounded-full">
                  <Scale className="w-6 h-6 text-blue-600" />
                </div>
                <h2 className="text-xl font-bold">Legal Classification</h2>
                <p className="text-gray-500">
                  Can models distinguish between a "Venta" (Sale) and a "Poder Especial" (Power of Attorney) in archaic Spanish?
                </p>
              </div>
              <div className="flex flex-col items-center space-y-4 text-center p-6 bg-white rounded-xl shadow-sm border">
                <div className="p-3 bg-purple-50 rounded-full">
                  <BarChart3 className="w-6 h-6 text-purple-600" />
                </div>
                <h2 className="text-xl font-bold">Cross-Lingual Bias</h2>
                <p className="text-gray-500">
                  Does translating 17th-century concepts into modern English degrade legal reasoning? We measure the "Translation Delta".
                </p>
              </div>
              <div className="flex flex-col items-center space-y-4 text-center p-6 bg-white rounded-xl shadow-sm border">
                <div className="p-3 bg-amber-50 rounded-full">
                  <BookOpen className="w-6 h-6 text-amber-600" />
                </div>
                <h2 className="text-xl font-bold">Digital Archive</h2>
                <p className="text-gray-500">
                  Access the SANRLite dataset: 1,300+ notarial records from 1653-1658 Buenos Aires, fully transcribed.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="py-6 w-full shrink-0 items-center px-4 md:px-6 border-t">
        <p className="text-xs text-gray-500 text-center">
          © 2025 SANR-Embed Project. Released under MIT License.
        </p>
      </footer>
    </div>
  );
}
