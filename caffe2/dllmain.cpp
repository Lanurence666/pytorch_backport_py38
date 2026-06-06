// DllInitMarker for torch_cpu.dll
// Uses init_seg(lib) to ensure this global constructor runs before all other
// global constructors in torch_cpu.dll. This sets a flag in c10.dll to suppress
// TORCH_CHECK exceptions during DLL initialization, preventing Error 1114
// (DLL initialization routine failed) on Windows.
//
// On Windows, global constructors run under the loader lock (inside DllMain).
// If a TORCH_CHECK fails during this time, the thrown C++ exception cannot
// propagate through the loader lock, causing the OS to return Error 1114.

#pragma warning(push)
#pragma warning(disable: 4073) // warning C4073: initializers put in library initialization area
#pragma init_seg(lib)

extern "C" __declspec(dllimport) void markDllInitBegin();
extern "C" __declspec(dllimport) void markDllInitEnd();

struct DllInitMarker {
  DllInitMarker() {
    markDllInitBegin();
  }
  ~DllInitMarker() {
    markDllInitEnd();
  }
};

// This global object is constructed before all other global objects in this DLL
// (due to init_seg(lib)), and destructed after all other global objects.
static DllInitMarker g_dllInitMarker;

#pragma warning(pop)
