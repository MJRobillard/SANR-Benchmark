'use client';

import { DocumentRecord } from "@/types/schema";
import { useState, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface EmbeddingPlotProps {
  documents: DocumentRecord[];
}

// Mock function to simulate 2D coordinates from embeddings
// In a real scenario, this would come from a pre-calculated UMAP/t-SNE CSV
const generateMockCoordinates = (doc: DocumentRecord) => {
  // Use ID to deterministically generate random-looking coords
  const seed = parseInt(doc.id) || 0;
  const x = (seed % 100) + (doc.year === 1653 ? -50 : 50) + (Math.random() * 10);
  const y = (seed % 73) + (doc.notary.length * 2) + (Math.random() * 10);
  return { x, y };
};

export default function EmbeddingPlot({ documents }: EmbeddingPlotProps) {
  const [colorMode, setColorMode] = useState<'year' | 'notary' | 'class'>('class');
  const [hoveredDoc, setHoveredDoc] = useState<DocumentRecord | null>(null);

  const dataPoints = useMemo(() => {
    return documents.map(doc => ({
      ...doc,
      ...generateMockCoordinates(doc)
    }));
  }, [documents]);

  const getColor = (doc: DocumentRecord) => {
    if (colorMode === 'year') return doc.year === 1653 ? '#3b82f6' : '#ef4444'; // Blue (1653) vs Red (1658)
    if (colorMode === 'class') {
        // Simple hash for color
        const hash = doc.label_primary?.split('').reduce((a,b) => a+b.charCodeAt(0), 0) || 0;
        return `hsl(${hash % 360}, 70%, 50%)`;
    }
    // Notary
    const hash = doc.notary?.split('').reduce((a,b) => a+b.charCodeAt(0), 0) || 0;
    return `hsl(${hash % 360}, 60%, 45%)`;
  };

  return (
    <div className="flex flex-col h-[600px] w-full border rounded-lg overflow-hidden bg-neutral-50 relative">
      
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 bg-white p-2 rounded-md shadow-sm border flex gap-2">
        <span className="text-xs font-medium self-center mr-2">Color By:</span>
        <Button 
            size="sm" 
            variant={colorMode === 'class' ? 'default' : 'outline'}
            onClick={() => setColorMode('class')}
        >
            Class
        </Button>
        <Button 
            size="sm" 
            variant={colorMode === 'year' ? 'default' : 'outline'}
            onClick={() => setColorMode('year')}
        >
            Year
        </Button>
        <Button 
            size="sm" 
            variant={colorMode === 'notary' ? 'default' : 'outline'}
            onClick={() => setColorMode('notary')}
        >
            Notary
        </Button>
      </div>

      {/* Canvas Area (SVG for simplicity in this mock) */}
      <div className="flex-1 w-full h-full p-8">
        <svg className="w-full h-full overflow-visible" viewBox="0 0 200 200">
            {dataPoints.map((point) => (
                <circle
                    key={point.id}
                    cx={point.x}
                    cy={point.y}
                    r={hoveredDoc?.id === point.id ? 4 : 2}
                    fill={getColor(point)}
                    opacity={hoveredDoc && hoveredDoc.id !== point.id ? 0.3 : 0.8}
                    className="cursor-pointer transition-all duration-200"
                    onMouseEnter={() => setHoveredDoc(point)}
                    onMouseLeave={() => setHoveredDoc(null)}
                />
            ))}
        </svg>
      </div>

      {/* Tooltip Overlay */}
      {hoveredDoc && (
          <div className="absolute bottom-4 left-4 z-20 w-80">
              <Card>
                  <div className="p-4 space-y-2">
                      <div className="flex justify-between">
                          <Badge variant="outline">ID: {hoveredDoc.id}</Badge>
                          <span className="text-xs text-muted-foreground">{hoveredDoc.year}</span>
                      </div>
                      <h4 className="font-bold text-sm">{hoveredDoc.label_primary}</h4>
                      <p className="text-xs text-muted-foreground line-clamp-2">{hoveredDoc.text_original}</p>
                      <p className="text-xs font-medium text-blue-600">{hoveredDoc.notary}</p>
                  </div>
              </Card>
          </div>
      )}
    </div>
  );
}






