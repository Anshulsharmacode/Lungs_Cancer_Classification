"use client"
import { FaArrowRight, FaLungsVirus, FaDna, FaChartLine, FaHeartbeat, FaStethoscope } from 'react-icons/fa';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  const handleStartAnalysis = () => {
    router.push('/result');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 relative overflow-hidden">
      <div className="absolute -right-40 -top-40 w-96 h-96 bg-blue-200 opacity-10 rounded-full blur-3xl" />
      <div className="absolute -left-40 -bottom-40 w-96 h-96 bg-indigo-200 opacity-10 rounded-full blur-3xl" />

      <div className="container mx-auto px-6 py-16 relative">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Content Section */}
          <div className="space-y-10 relative">
            <div className="space-y-4">
              <div className="inline-block px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-full mb-4">
                <span className="text-xs font-medium tracking-wider">MAJOR PROJECT 2023-24</span>
              </div>
              <h1 className="text-4xl font-bold text-gray-800 leading-tight">
                Lung Cancer
                <span className="block mt-2 text-5xl bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                  Detection System
                </span>
              </h1>
              <p className="text-sm text-gray-600 max-w-xl leading-relaxed">
                An advanced medical imaging analysis system using deep learning for accurate lung cancer detection
              </p>
            </div>

            {/* Feature Cards */}
            <div className="space-y-3">
              <div className="p-4 bg-white/80 backdrop-blur-sm border-l-4 border-blue-600 rounded-lg shadow-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center space-x-3">
                  <FaDna className="text-2xl text-blue-600" />
                  <div>
                    <h3 className="text-base font-semibold text-gray-800">AI-Powered Analysis</h3>
                    <p className="text-xs text-gray-500">Deep learning model for tumor detection</p>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-white/80 backdrop-blur-sm border-l-4 border-indigo-600 rounded-lg shadow-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center space-x-3">
                  <FaChartLine className="text-2xl text-indigo-600" />
                  <div>
                    <h3 className="text-base font-semibold text-gray-800">Medical Imaging</h3>
                    <p className="text-xs text-gray-500">Advanced CT scan processing system</p>
                  </div>
                </div>
              </div>
            </div>

            {/* CTA Button */}
            <button
              onClick={handleStartAnalysis}
              className="group inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 shadow-md hover:shadow-lg"
            >
              Start Analysis
              <FaArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" />
            </button>
          </div>

          {/* Right Side Visualization - Desktop only */}
          <div className="hidden lg:block relative">
            <div className="relative z-20 group">
              {/* Main Icon */}
              <div className="relative transform transition-all duration-700 hover:scale-105">
                <FaLungsVirus className="text-blue-900/30 w-96 h-96 animate-float" />
                
                {/* Orbiting Icons */}
                <div className="absolute inset-0 animate-spin-slow">
                  <FaHeartbeat className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-indigo-600/40 w-12 h-12" />
                  <FaStethoscope className="absolute top-1/2 -right-8 transform -translate-y-1/2 text-blue-600/40 w-12 h-12" />
                  <FaDna className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-indigo-600/40 w-12 h-12" />
                  <FaChartLine className="absolute top-1/2 -left-8 transform -translate-y-1/2 text-blue-600/40 w-12 h-12" />
                </div>

                {/* Gradient Overlays */}
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-indigo-500/10 rounded-full blur-3xl group-hover:blur-2xl transition-all duration-700" />
                <div className="absolute inset-0 bg-gradient-to-tl from-transparent via-white/50 to-transparent rounded-full" />
              </div>

              {/* Pulse Effect */}
              <div className="absolute inset-0 animate-ping-slow opacity-20">
                <div className="absolute inset-0 bg-blue-200 rounded-full blur-2xl transform scale-90" />
              </div>
            </div>
          </div>
        </div>

        {/* Mobile Icon - Moved to bottom */}
        <div className="block lg:hidden relative mt-16">
          <div className="relative z-20 group">
            <div className="relative transform transition-all duration-700">
              <FaLungsVirus className="text-blue-900/30 w-48 h-48 mx-auto animate-float" />
              
              {/* Orbiting Icons - Smaller for mobile */}
              <div className="absolute inset-0 animate-spin-slow">
                <FaHeartbeat className="absolute -top-4 left-1/2 transform -translate-x-1/2 text-indigo-600/40 w-8 h-8" />
                <FaStethoscope className="absolute top-1/2 -right-4 transform -translate-y-1/2 text-blue-600/40 w-8 h-8" />
                <FaDna className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-indigo-600/40 w-8 h-8" />
                <FaChartLine className="absolute top-1/2 -left-4 transform -translate-y-1/2 text-blue-600/40 w-8 h-8" />
              </div>

              {/* Mobile Gradient Overlays */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-indigo-500/10 rounded-full blur-2xl" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}