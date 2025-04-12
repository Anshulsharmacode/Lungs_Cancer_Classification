import React from 'react';

const Navbar = () => {
  return (
    <header className="bg-blue-600 text-white py-4">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-2xl font-bold">Major Project 2024-25</h1>
        <nav>
          <ul className="flex space-x-4">
            <li><a href="/" className="hover:underline">Home</a></li>
            <li><a href="/result" className="hover:underline">Results</a></li>
            {/* <li><a href="/about" className="hover:underline">About</a></li> */}
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
