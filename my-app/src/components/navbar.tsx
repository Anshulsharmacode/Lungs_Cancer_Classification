import React from 'react';
import Link from 'next/link';
import { theme } from '@/components/theme';

const Navbar = () => {
  return (
    <header style={{ backgroundColor: theme.colors.primary.main }} className="text-white py-4">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-2xl font-bold">Major Project 2024-25</h1>
        <nav>
          <ul className="flex space-x-4">
            <li><Link href="/" className="hover:underline">Home</Link></li>
            <li><Link href="/result" className="hover:underline">Results</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
