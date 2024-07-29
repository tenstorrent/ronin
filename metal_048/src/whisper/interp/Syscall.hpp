// Copyright 2020 Western Digital Corporation or its affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>



namespace WdRiscv
{
  template <typename URV>
  class Hart;


  template <typename URV>
  class Syscall
  {
  public:
    
    /// Signed register type corresponding to URV. For example, if URV
    /// is uint32_t, then SRV will be int32_t.
    typedef typename std::make_signed_t<URV> SRV;

    Syscall(Hart<URV>& hart)
      : hart_(hart)
    {
    	auto mem_size = hart.getMemorySize();
    	mmap_blocks_.insert(std::make_pair(mem_size/2L, blk_t(mem_size/2L, true)));
    }
//    void print_mmap(const std::string prefix) {
//    	 for(auto& it: mmap_blocks_)
//    		 printf("%s --> 0x%llx: 0x%llx, %d\n",prefix.c_str(), it.first, it.second.length, it.second.free);
//    	 fflush(stdout);
//
//    }
    /// Emulate a system call on the associated hart. Return an integer
    /// value corresponding to the result.
    URV emulate();

    /// Redirect the given output file descriptor (typically stdout or
    /// stderr) to the given file. Return true on success and false on
    /// failure.
    bool redirectOutputDescriptor(int fd, const std::string& path);

    void enableLinux(bool flag)
    { linux_ = flag; }
      
    /// Save the currently open file descriptors to the given file.
    bool saveFileDescriptors(const std::string& path);

    /// Load and open the file descriptors previously saved in given file.
    bool loadFileDescriptors(const std::string& path);

    /// Report the files opened by the target RISCV program during
    /// current run.
    void reportOpenedFiles(std::ostream& out);

    uint64_t mmap_alloc(uint64_t size);

    int mmap_dealloc(uint64_t addr, uint64_t size);

    uint64_t mmap_remap(uint64_t addr, uint64_t old_size, uint64_t new_size, bool maymove);

    typedef std::pair<uint64_t, uint64_t> AddrLen;  // Address/length pair

    /// Clear memory changes recorded by last emulate call.
    void clearMemoryChanges()
    { memChanges_.clear(); }

    /// Copy the memory changes recorded by the last emulate call to the
    /// given vector.
    void getMemoryChanges(std::vector<AddrLen>& changes) const
    { changes = memChanges_; }

    void getUsedMemBlocks(std::vector<AddrLen>& used_blocks);

    bool loadUsedMemBlocks(const std::string& path, std::vector<AddrLen>& used_blocks);
    bool saveUsedMemBlocks(const std::string& path, std::vector<AddrLen>& used_blocks);

    bool saveMmap(const std::string & filename);

    bool loadMmap(const std::string & filename);

  protected:

    friend class Hart<URV>;

    /// For Linux emulation: Set initial target program break to the
    /// RISCV page address larger than or equal to the given address.
    void setTargetProgramBreak(URV addr)
    { progBreak_ = addr; }

    /// Return target program break.
    URV targetProgramBreak() const
    { return progBreak_; }

    /// Map Linux file descriptor to a RISCV file descriptor and install
    /// the result in the riscv-to-linux fd map. Return remapped
    /// descritpor or -1 if remapping is not possible.
    int registerLinuxFd(int linuxFd, const std::string& path, bool isRead);

    /// Return the effective (after redirection) file descriptor
    /// corresponding to the target program file descriptor.
    int effectiveFd(int fd)
    {
      if (fdMap_.count(fd))
        return fdMap_.at(fd);
      return fd;
    }

  private:

    Hart<URV>& hart_;
    bool linux_ = false;
    URV progBreak_ = 0;          // For brk Linux emulation.

    struct blk_t {
    	blk_t(uint64_t length, bool free): length(length), free(free) {};
    	uint64_t length;
    	bool free;
    };
    using blk_map_t = std::map<uint64_t, blk_t>;
    blk_map_t mmap_blocks_;

    std::unordered_map<int, int> fdMap_;
    std::unordered_map<int, bool> fdIsRead_;
    std::unordered_map<int, std::string> fdPath_;
    std::unordered_set<std::string> readPaths_;
    std::unordered_set<std::string> writePaths_;

    std::vector<AddrLen> memChanges_;  // Memory locations changed by syscall.

  };
}
