#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

namespace torch::distributed::rpc {
const std::string REMOTE_PROFILING_KEY_PREFIX = "#remote_op: ";
constexpr int kAutoIncrementBits = 48;

/*static */ RemoteProfilerManager& RemoteProfilerManager::getInstance() {
  static RemoteProfilerManager* handler = new RemoteProfilerManager();
  return *handler;
}

void RemoteProfilerManager::setCurrentKey(std::string key) {
  if (getCurrentThreadLocalKey()) {
    TORCH_CHECK(
        false,
        "Cannot call RemoteProfilerManager::setCurrentKey when current key is already set.");
  }
  setCurrentThreadLocalKey(std::move(key));
}

bool RemoteProfilerManager::isCurrentKeySet() const {
  return getCurrentThreadLocalKey().has_value();
}

void RemoteProfilerManager::unsetCurrentKey() {
  setCurrentThreadLocalKey(std::nullopt);
}

void RemoteProfilerManager::eraseKey(const ProfilingId& globallyUniqueId) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  profiledRpcKeys_.erase(it);
}

std::string RemoteProfilerManager::retrieveRPCProfilingKey(
    const ProfilingId& globallyUniqueId) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  return it->second;
}

ProfilingId RemoteProfilerManager::getNextProfilerId() {
  auto localId = getNextLocalId();
  auto localWorkerId = RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_;
  auto globallyUniqueId =
      torch::distributed::rpc::ProfilingId(localWorkerId, localId);
  return globallyUniqueId;
}

local_id_t RemoteProfilerManager::getNextLocalId() {
  std::lock_guard<std::mutex> guard(mutex_);
  return currentLocalId_++;
}

std::string& RemoteProfilerManager::getCurrentProfilingKey() {
  TORCH_CHECK(
      getCurrentThreadLocalKey(),
      "Must set currentThreadLocalKey_ before calling getCurrentProfilingKey");
  return *getCurrentThreadLocalKey();
}

void RemoteProfilerManager::saveRPCKey(
    ProfilingId globallyUniqueId,
    const std::string& rpcProfilingKey) {
  std::lock_guard<std::mutex> guard(mutex_);
  profiledRpcKeys_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(globallyUniqueId),
      std::forward_as_tuple(rpcProfilingKey));
}

RemoteProfilerManager::RemoteProfilerManager() {
  auto workerId =
      static_cast<int64_t>(RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_);
  currentLocalId_ = workerId << kAutoIncrementBits;
}
} // namespace torch::distributed::rpc
