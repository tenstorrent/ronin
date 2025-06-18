// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "elfio/elfio.hpp"

#include "linker/linker.hpp"

namespace riscv {
namespace linker {

using Elfio = ELFIO::elfio;
using Sections = ELFIO::elfio::Sections; 
using Section = ELFIO::section;

class LinkerImpl: public Linker {
public:
    LinkerImpl();
    ~LinkerImpl();
public:
    void add_builtin(const std::string &name, uint64_t value) override;
    void link(
        const std::string &fname, 
        uint64_t code_base,
        std::vector<uint8_t> &result,
        uint64_t &start_pc) override;
private:
    void reset();
    void load_elf_file(const std::string &fname);
    void collect_sections();
    void collect_code_sections();
    void collect_symbol_sections();
    void collect_reloc_sections();
    void collect_null_sections();
    void enter_code_section(int index, Section *section);
    void enter_symbol_section(int index, Section *section);
    void enter_reloc_section(int index, Section *section);
    void resolve_builtins();
    void validate_builtin(const std::string &name, int bind, int section_index);
    void relocate();
    void init_code();
    uint64_t relocate_entry(
        uint64_t reloc_offset,
        uint64_t offset,
        const std::string &symbol_name,
        int type,
        int addend);
    bool get_symbol_value(const std::string &name, uint64_t &value);
    void reloc_b_type(uint64_t offset, uint64_t value);
    void reloc_cb_type(uint64_t offset, uint64_t value);
    void reloc_cj_type(uint64_t offset, uint64_t value);
    void reloc_i_type(uint64_t offset, uint64_t value);
    void reloc_u_type(uint64_t offset, uint64_t value);
    void reloc_ui_type(uint64_t offset, uint64_t value);
    void move_code(std::vector<uint8_t> &result);
    uint64_t get_start_pc();
    bool map_builtin(const std::string &name, uint64_t &offset);
    void bind_symbol(const std::string &name, int local_index, int entry_index);
    bool map_symbol(const std::string &name, int &local_index, int &entry_index);
    void bind_symbol_section(int index, int local_index);
    bool map_symbol_section(int index, int &local_index);
    void bind_code_section(int index, int local_index);
    bool map_code_section(int index, int &local_index);
    void check_code_offset(uint64_t offset, uint64_t size);
    void diag_reloc_entry(uint64_t offset, uint64_t value);
    void diag_reloc_code(const char *tag, uint64_t offset);
    void diag_reloc_code_c(const char *tag, uint64_t offset);
private:
    struct SymbolEntry {
        int orig_index;
        std::string name;
        uint64_t value;
        uint64_t size;
        int bind;
        int type;
        int section_index;
        bool is_builtin;
    };
    struct SymbolSection {
        int orig_index;
        std::string name;
        std::vector<SymbolEntry> entries;
    };
    struct RelocEntry {
        int orig_index;
        uint64_t offset;
        uint64_t symbol_value;
        std::string symbol_name;
        int type;
        int addend;
        uint64_t calc_value;
    };
    struct RelocSection {
        int orig_index;
        std::string name;
        int info;
        std::vector<RelocEntry> entries;
    };
    struct CodeSection {
        int orig_index;
        std::string name;
        uint64_t size;
        uint64_t addr_align;
        const uint8_t *data;
        uint64_t offset;
    };
private:
    std::unordered_map<std::string, uint64_t> m_builtin_map;
    uint64_t m_code_base;
    Elfio m_elfio;
    std::vector<SymbolSection> m_symbol_sections;
    std::vector<RelocSection> m_reloc_sections;
    std::vector<CodeSection> m_code_sections;
    std::unordered_map<std::string, std::pair<int, int>> m_symbol_map;
    std::unordered_map<int, int> m_symbol_section_map;
    std::unordered_map<int, int> m_code_section_map;
    std::unordered_set<int> m_null_sections;
    std::unordered_map<uint64_t, uint32_t> m_pcrel_lo_map;
    uint64_t m_code_end;
    std::vector<uint8_t> m_code;
};

} // linker
} // riscv

