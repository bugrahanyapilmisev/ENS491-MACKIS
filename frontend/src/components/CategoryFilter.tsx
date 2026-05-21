import { categories } from "../lib/mockData";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";

interface CategoryFilterProps {
  selected: string;
  onSelect: (category: string) => void;
}

export function CategoryFilter({ selected, onSelect }: CategoryFilterProps) {
  return (
    <ScrollArea className="w-full">
      <div className="flex gap-2 p-4 border-b">
        {categories.map((category) => (
          <Button
            key={category.id}
            variant={selected === category.id ? "default" : "outline"}
            size="sm"
            onClick={() => onSelect(category.id)}
            className="shrink-0"
          >
            <span className="mr-1.5">{category.icon}</span>
            {category.label}
          </Button>
        ))}
      </div>
    </ScrollArea>
  );
}
