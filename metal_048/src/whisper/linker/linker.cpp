
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>

#include "elfio/elfio.hpp"

#include "linker/linker.hpp"
#include "linker/private.hpp"

namespace riscv {
namespace linker {

using ELFIO::Elf_Half;
using ELFIO::Elf_Word;
using ELFIO::Elf_Sword;
using ELFIO::Elf_Xword;
using ELFIO::Elf_Sxword;

using ELFIO::Elf32_Addr;
using ELFIO::Elf32_Off;
using ELFIO::Elf64_Addr;
using ELFIO::Elf64_Off;

namespace {

const bool DIAG_RELOC_ENABLED = false;

uint64_t align(uint64_t offset, uint64_t bytes) {
    if (bytes == 0 || bytes == 1) {
        return offset;
    } else {
        return ((offset + bytes - 1) / bytes) * bytes;
    }
}

void copy_data(uint8_t *dst, const uint8_t *src, uint64_t size) {
    memcpy(dst, src, size_t(size));
}

void check(bool succ, const std::string &msg) {
    if (!succ) {
        throw std::runtime_error(msg);
    }
}

void check_signed_range(uint64_t value, int bits) {
    uint64_t mask = (~uint64_t(0)) << (bits - 1);
    uint64_t test = value & mask;
    if (test != 0 && test != mask) {
        std::string msg = 
            "Value " + std::to_string(value) + 
                " is out of signed " + std::to_string(bits) + "-bit range";
        throw std::runtime_error(msg);
    }
}

//
//    Relocation types: processor-specific, see 
//    https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc
//

enum class RelocType {
    NONE = 0,
    _32 = 1,
    _64 = 2,
    RELATIVE = 3,      // dynamic
    COPY = 4,          // dynamic
    JUMP_SLOT = 5,     // dynamic
    TLS_DTPMOD32 = 6,  // dynamic
    TLS_DTPMOD64 = 7,  // dynamic
    TLS_DTPREL32 = 8,  // dynamic
    TLS_DTPREL64 = 9,  // dynamic
    TLS_TPREL32 = 10,  // dynamic
    TLS_TPREL64 = 11,  // dynamic
    BRANCH = 16,
    JAL = 17,
    CALL = 18,
    CALL_PLT = 19,
    GOT_HI20 = 20,
    TLS_GOT_HI20 = 21,
    TLS_GD_HI20 = 22,
    PCREL_HI20 = 23,
    PCREL_LO12_I = 24,
    PCREL_LO12_S = 25,
    HI20 = 26,
    LO12_I = 27,
    LO12_S = 28,
    TPREL_HI20 = 29,
    TPREL_LO12_I = 30,
    TPREL_LO12_S = 31,
    TPREL_ADD = 32,
    ADD8 = 33,
    ADD16 = 34,
    ADD32 = 35,
    ADD64 = 36,
    SUB8 = 37,
    SUB16 = 38,
    SUB32 = 39,
    SUB64 = 40,
    ALIGN = 43,
    RVC_BRANCH = 44,
    RVC_JUMP = 45,
    RVC_LUI = 46,
    RELAX = 51,
    SUB6 = 52,
    SET6 = 53,
    SET8 = 54,
    SET16 = 55,
    SET32 = 56,
    _32_PCREL = 57,
    IRELATIVE = 68,    // dynamic
    PLT32 = 59,
    SET_ULEB128 = 60,
    SUB_ULEB128 = 61
};

void bad_reloc_type(RelocType reloc_type) {
    throw std::runtime_error(
        "Invalid or unsupported relocation type " + std::to_string(int(reloc_type)));
}

void split_pcrel_hi_lo(uint32_t value, uint32_t &hi20, uint32_t &lo12) {
    lo12 = value & 0xfff;
    if ((lo12 & 0x800) != 0) {
        // represent as signed to compute hi20
        lo12 |= 0xfffff000;
    }
    hi20 = (value - lo12) >> 12;
}

} // namespace

//
//    Linker
//

Linker *Linker::create() {
    return new LinkerImpl();
}

//
//    LinkerImpl
//

LinkerImpl::LinkerImpl():
        m_code_base(0),
        m_code_end(0) { }

LinkerImpl::~LinkerImpl() { }

void LinkerImpl::add_builtin(const std::string &name, uint64_t value) {
    m_builtin_map.emplace(name, value);
}

void LinkerImpl::link(
        const std::string &fname, 
        uint64_t code_base,
        std::vector<uint8_t> &result,
        uint64_t &start_pc) {
    m_code_base = code_base;
    reset();
    load_elf_file(fname);
    collect_sections();
    resolve_builtins();
    relocate();
    move_code(result);
    start_pc = get_start_pc();
}

void LinkerImpl::reset() {
    m_symbol_sections.clear();
    m_reloc_sections.clear();
    m_code_sections.clear();
    m_symbol_map.clear();
    m_symbol_section_map.clear();
    m_code_section_map.clear();
    m_null_sections.clear();
    m_pcrel_lo_map.clear();
    m_code_end = m_code_base;
    m_code.clear();
}

void LinkerImpl::load_elf_file(const std::string &fname) {
    bool succ = m_elfio.load(fname);
    check(succ, "Cannot load ELF file " + fname);
}

void LinkerImpl::collect_sections() {
    collect_code_sections();
    collect_symbol_sections();
    collect_reloc_sections();
    collect_null_sections();
}

void LinkerImpl::collect_code_sections() {
    Sections &sections = m_elfio.sections; 
    int sections_size = int(sections.size());
    for (int i = 0; i < sections_size; i++) {
        Section *section = sections[i];
        if (section->get_type() == SHT_PROGBITS) {
            enter_code_section(i, section);
        }
    }
}

void LinkerImpl::collect_symbol_sections() {
    Sections &sections = m_elfio.sections; 
    int sections_size = int(sections.size());
    for (int i = 0; i < sections_size; i++) {
        Section *section = sections[i];
        if (section->get_type() == SHT_SYMTAB) {
            enter_symbol_section(i, section);
        }
    }
}

void LinkerImpl::collect_reloc_sections() {
    Sections &sections = m_elfio.sections; 
    int sections_size = int(sections.size());
    for (int i = 0; i < sections_size; i++) {
        Section *section = sections[i];
        if (section->get_type() == SHT_RELA) {
            enter_reloc_section(i, section);
        }
    }
}

void LinkerImpl::collect_null_sections() {
    Sections &sections = m_elfio.sections; 
    int sections_size = int(sections.size());
    for (int i = 0; i < sections_size; i++) {
        Section *section = sections[i];
        if (section->get_type() == SHT_NULL) {
            m_null_sections.insert(i);
        }
    }
}

void LinkerImpl::enter_code_section(int index, Section *section) {
    int loc_index = int(m_code_sections.size());
    m_code_sections.emplace_back();
    CodeSection &cs = m_code_sections[loc_index];
    cs.orig_index = index;
    cs.name = section->get_name();
    cs.size = uint64_t(section->get_size());
    cs.addr_align = uint64_t(section->get_addr_align());
    cs.data = reinterpret_cast<const uint8_t *>(section->get_data());
    cs.offset = align(m_code_end, cs.addr_align);
    m_code_end = cs.offset + cs.size;
    bind_code_section(index, loc_index);
}

void LinkerImpl::enter_symbol_section(int index, Section *section) {
    int local_index = int(m_symbol_sections.size());
    m_symbol_sections.emplace_back();
    SymbolSection &ss = m_symbol_sections[local_index];
    ss.orig_index = index;
    ss.name = section->get_name();
    ELFIO::symbol_section_accessor accessor(m_elfio, section);
    int symbols_num = int(accessor.get_symbols_num());
    for (int i = 0; i < symbols_num; i++) {
        std::string name;
        Elf64_Addr value;
        Elf_Xword size;
        unsigned char bind;
        unsigned char type;
        Elf_Half section_index;
        unsigned char other;
        bool succ =
            accessor.get_symbol(
                i,
                name,
                value,
                size,
                bind,
                type,
                section_index,
                other);
        check(succ, "Symbol section accessor get_symbol error");
        if (name.empty()) {
            // skip unnamed symbols (are they of any use?)
            continue;
        }
        int entry_index = int(ss.entries.size());
        ss.entries.emplace_back();
        SymbolEntry &entry = ss.entries[entry_index];
        entry.orig_index = i;
        entry.name = name;
        entry.value = uint32_t(value);
        entry.size = uint32_t(size);
        entry.bind = int(bind);
        entry.type = int(type);
        entry.section_index = int(section_index);
        entry.is_builtin = false; // will be resolved later
        bind_symbol(name, local_index, entry_index);
    }
    bind_symbol_section(index, local_index);
}

void LinkerImpl::enter_reloc_section(int index, Section *section) {
    int loc_index = int(m_reloc_sections.size());
    m_reloc_sections.emplace_back();
    RelocSection &rs = m_reloc_sections[loc_index];
    rs.orig_index = index;
    rs.name = section->get_name();
    rs.info = int(section->get_info());
    ELFIO::relocation_section_accessor accessor(m_elfio, section);
    int entries_num = int(accessor.get_entries_num());
    for (int i = 0; i < entries_num; i++) {
        Elf64_Addr offset;
        Elf64_Addr symbol_value;
        std::string symbol_name;
        Elf_Word type;
        Elf_Sxword addend;
        Elf_Sxword calc_value;
        bool succ = 
            accessor.get_entry(
                i, 
                offset, 
                symbol_value, 
                symbol_name, 
                type, 
                addend, 
                calc_value);
        check(succ, "Relocation section accessor get_entry error");
        if (symbol_name.empty()) {
            // skip unnamed symbols (are they of any use?)
            continue;
        }
        int entry_index = int(rs.entries.size());
        rs.entries.emplace_back();
        RelocEntry &entry = rs.entries[entry_index];
        entry.orig_index = i;
        entry.offset = uint64_t(offset);
        entry.symbol_value = uint64_t(symbol_value);
        entry.symbol_name = symbol_name;
        entry.type = int(type);
        entry.addend = int(addend);
        entry.calc_value = 0; // will be computed at relocation phase
    }
}

void LinkerImpl::resolve_builtins() {
    for (SymbolSection &ss: m_symbol_sections) {
        for (SymbolEntry &entry: ss.entries) {
            uint64_t value;
            if (!map_builtin(entry.name, value)) {
//printf("@@@ Unmapped builtin [%s]\n", entry.name.c_str());
                continue;
            }
            validate_builtin(entry.name, entry.bind, entry.section_index);
            entry.value = value;
            entry.is_builtin = true;
        }
    }
}

void LinkerImpl::validate_builtin(const std::string &name, int bind, int section_index) {
    if (bind != STB_GLOBAL) {
        throw std::runtime_error("Builtin symbol " + name + " must have global bind");
    }
    if (m_null_sections.count(section_index) == 0) {
        throw std::runtime_error("Builtin symbol " + name + " must be related with null section");
    }
}

void LinkerImpl::relocate() {
    m_pcrel_lo_map.clear();
    init_code();
    for (RelocSection &rs: m_reloc_sections) {
        int cs_index;
        if (!map_code_section(rs.info, cs_index)) {
            throw std::runtime_error("Undefined code section " + std::to_string(rs.info));
        }
        CodeSection &cs = m_code_sections[cs_index];
        for (RelocEntry &entry: rs.entries) {
            entry.calc_value =
                relocate_entry(
                    cs.offset,
                    entry.offset,
                    entry.symbol_name,
                    entry.type,
                    entry.addend);
        }
    }
}

void LinkerImpl::init_code() {
    m_code.resize(m_code_end - m_code_base, 0);
    uint8_t *code_ptr = m_code.data();
    for (CodeSection &cs: m_code_sections) {
        copy_data(code_ptr + cs.offset - m_code_base, cs.data, cs.size);
    }
}

uint64_t LinkerImpl::relocate_entry(
        uint64_t reloc_offset,
        uint64_t offset,
        const std::string &symbol_name,
        int type,
        int addend) {
    uint64_t symbol_value = 0;
    if (!get_symbol_value(symbol_name, symbol_value)) {
        // is this legal?
        return 0;
    }
    // A = Addend field in the relocation entry associated with the symbol
    // P = Position of the relocation
    // S = Value of the symbol in the symbol table
    // P and S are global (include respective section offset)
    uint64_t var_a = uint64_t(addend);
    uint64_t var_p = reloc_offset + offset;
    uint64_t var_s = symbol_value;
    uint64_t calc_value = 0;
    RelocType reloc_type = RelocType(type);
    switch (reloc_type) {
    case RelocType::BRANCH:
        // S + A - P: B-Type
        {
            calc_value = var_s + var_a - var_p;
            reloc_b_type(var_p, calc_value);
        }
        break;
    case RelocType::CALL:
    case RelocType::CALL_PLT:
        // S + A - P: U+I-Type
        {
            calc_value = var_s + var_a - var_p;
            reloc_ui_type(var_p, calc_value);
        }
        break;
    case RelocType::PCREL_HI20:
        // S + A - P: U-Type
        {
            calc_value = var_s + var_a - var_p;
            check_signed_range(calc_value, 32);
            uint32_t hi20, lo12;
            split_pcrel_hi_lo(uint32_t(calc_value), hi20, lo12);
            reloc_u_type(var_p, uint64_t(hi20));
            m_pcrel_lo_map.emplace(var_p, lo12); 
        }
        break;
    case RelocType::PCREL_LO12_I:
        // S - P: I-Type
        {
            auto it = m_pcrel_lo_map.find(var_s);
            if (it == m_pcrel_lo_map.end()) {
                throw std::runtime_error("Undefined pcrel for symbol " + symbol_name);
            }
            uint32_t lo12 = it->second;
            reloc_i_type(var_p, uint64_t(lo12));
        }
        break;
    case RelocType::HI20:
        // S + A: U-Type
        {
            calc_value = var_s + var_a;
            reloc_u_type(var_p, (calc_value >> 12));
        }
        break;
    case RelocType::LO12_I:
        // S + A: I-Type
        {
            calc_value = var_s + var_a;
            reloc_i_type(var_p, (calc_value & 0xfff));
        }
        break;
    case RelocType::RVC_BRANCH:
        // S + A - P: CB-Type
        {
            calc_value = var_s + var_a - var_p;
            reloc_cb_type(var_p, calc_value);
        }
        break;
    case RelocType::RVC_JUMP:
        // S + A - P: CJ-Type
        {
            calc_value = var_s + var_a - var_p;
            reloc_cj_type(var_p, calc_value);
        }
        break;
    default:
        bad_reloc_type(reloc_type);
        break;
    }
    return calc_value;
}

bool LinkerImpl::get_symbol_value(const std::string &name, uint64_t &value) {
    int local_index;
    int entry_index;
    if (!map_symbol(name, local_index, entry_index)) {
        throw std::runtime_error("Undefined symbol " + name);
    }
    SymbolEntry &entry = m_symbol_sections[local_index].entries[entry_index];
    if (entry.is_builtin) {
        value = entry.value;
        return true;
    }
    int cs_index;
    if (!map_code_section(entry.section_index, cs_index)) {
        // symbol does not refer to code section: is this legal?
        value = 0;
        return false;
    }
    CodeSection &cs = m_code_sections[cs_index];
    value = cs.offset + entry.value;
    return true;
}

void LinkerImpl::reloc_b_type(uint64_t offset, uint64_t value) {
    // B-Type: Specifies a field as the immediate field in a B-type instruction
    check_code_offset(offset, 4);
    check_signed_range(value, 13);
    offset -= m_code_base;
    diag_reloc_entry(offset, value);
    diag_reloc_code("B-Type before", offset);
    uint8_t imm_12 = uint8_t((value >> 12) & 0x1);
    uint8_t imm_10_5 = uint8_t((value >> 5) & 0x3f);
    uint8_t imm_4_1 = uint8_t((value >> 1) & 0xf);
    uint8_t imm_11 = uint8_t((value >> 11) & 0x1);
    m_code[offset + 3] &= 0x1;
    m_code[offset + 3] |= ((imm_12 << 7) | (imm_10_5 << 1));
    m_code[offset + 1] &= 0xf0;
    m_code[offset + 1] |= imm_4_1;
    m_code[offset] &= 0x7f;
    m_code[offset] |= (imm_11 << 7);
    diag_reloc_code("B-Type after", offset);
}

void LinkerImpl::reloc_cb_type(uint64_t offset, uint64_t value) {
    // CB-Type: Specifies a field as the immediate field in a CB-type instruction
    check_code_offset(offset, 2);
    check_signed_range(value, 9);
    offset -= m_code_base;
    diag_reloc_entry(offset, value);
    diag_reloc_code_c("CB-Type before", offset);
    uint8_t imm_8 = uint8_t((value >> 8) & 0x1);
    uint8_t imm_4_3 = uint8_t((value >> 3) & 0x3);
    uint8_t imm_7_6 = uint8_t((value >> 6) & 0x3);
    uint8_t imm_2_1 = uint8_t((value >> 1) & 0x3);
    uint8_t imm_5 = uint8_t((value >> 5) & 0x1);
    m_code[offset + 1] &= 0xe3;
    m_code[offset + 1] |= ((imm_8 << 4) | (imm_4_3 << 2));
    m_code[offset] &= 0x83;
    m_code[offset] |= ((imm_7_6 << 5) | (imm_2_1 << 3) | (imm_5 << 2));
    diag_reloc_code_c("CB-Type after", offset);
}

void LinkerImpl::reloc_cj_type(uint64_t offset, uint64_t value) {
    // CJ-Type: Specifies a field as the immediate field in a CJ-type instruction
    check_code_offset(offset, 2);
    check_signed_range(value, 12);
    offset -= m_code_base;
    diag_reloc_entry(offset, value);
    diag_reloc_code_c("CJ-Type before", offset);
    uint8_t imm_11 = uint8_t((value >> 11) & 0x1);
    uint8_t imm_4 = uint8_t((value >> 4) & 0x1);
    uint8_t imm_9_8 = uint8_t((value >> 8) & 0x3);
    uint8_t imm_10 = uint8_t((value >> 10) & 0x1);
    uint8_t imm_6 = uint8_t((value >> 6) & 0x1);
    uint8_t imm_7 = uint8_t((value >> 7) & 0x1);
    uint8_t imm_3_1 = uint8_t((value >> 1) & 0x7);
    uint8_t imm_5 = uint8_t((value >> 5) & 0x1);
    m_code[offset + 1] &= 0xe0;
    m_code[offset + 1] |= ((imm_11 << 4) | (imm_4 << 3) | (imm_9_8 << 1) | imm_10);
    m_code[offset] &= 0x3;
    m_code[offset] |= ((imm_6 << 7) | (imm_7 << 6) | (imm_3_1 << 3) | (imm_5 << 2));
    diag_reloc_code_c("CJ-Type after", offset);
}

void LinkerImpl::reloc_i_type(uint64_t offset, uint64_t value) {
    // I-Type: Specifies a field as the immediate field in an I-type instruction
    check_code_offset(offset, 4);
    check_signed_range(value, 12);
    offset -= m_code_base;
    diag_reloc_entry(offset, value);
    diag_reloc_code("I-Type before", offset);
    uint8_t field = uint8_t((value >> 4) & 0xff);
    m_code[offset + 3] = field;
    field = uint8_t(value & 0xf);
    m_code[offset + 2] &= 0x0f;
    m_code[offset + 2] |= (field << 4);
    diag_reloc_code("I-Type after", offset);
}

void LinkerImpl::reloc_u_type(uint64_t offset, uint64_t value) {
    // U-Type: Specifies a field as the immediate field in an U-type instruction
    check_code_offset(offset, 4);
    check_signed_range(value, 20);
    offset -= m_code_base;
    diag_reloc_entry(offset, value);
    diag_reloc_code("U-Type before", offset);
    uint8_t field = uint8_t((value >> 12) & 0xff);
    m_code[offset + 3] = field;
    field = uint8_t((value >> 4) & 0xff);
    m_code[offset + 2] = field;
    field = uint8_t(value & 0xf);
    m_code[offset + 1] &= 0x0f;
    m_code[offset + 1] |= (field << 4);
    diag_reloc_code("U-Type after", offset);
}

void LinkerImpl::reloc_ui_type(uint64_t offset, uint64_t value) {
    // U+I-Type: Specifies a field as the immediate fields in a U-type and I-type instruction pair
    check_code_offset(offset, 8);
    check_signed_range(value, 32);
    offset -= m_code_base;
    diag_reloc_entry(offset, value);
    diag_reloc_code("U+I-Type before", offset);
    diag_reloc_code("U+I-Type before", offset + 4);
    uint32_t value20, value12;
    split_pcrel_hi_lo(uint32_t(value), value20, value12);
    // U-Type (20 upper bits)
    uint8_t field = uint8_t((value20 >> 12) & 0xff);
    m_code[offset + 3] = field;
    field = uint8_t((value20 >> 4) & 0xff);
    m_code[offset + 2] = field;
    field = uint8_t(value20 & 0xf);
    m_code[offset + 1] &= 0x0f;
    m_code[offset + 1] |= (field << 4);
    // I-Type (12 lower bits)
    field = uint8_t((value12 >> 4) & 0xff);
    m_code[offset + 7] = field;
    field = uint8_t(value12 & 0xf);
    m_code[offset + 6] &= 0x0f;
    m_code[offset + 6] |= (field << 4);
    diag_reloc_code("U+I-Type after", offset);
    diag_reloc_code("U+I-Type after", offset + 4);
}

void LinkerImpl::move_code(std::vector<uint8_t> &result) {
    result.clear();
    result.swap(m_code);
}

uint64_t LinkerImpl::get_start_pc() {
    uint64_t value;
    bool succ = get_symbol_value("main", value);
    if (!succ) {
        throw std::runtime_error("Undefined symbol 'main'");
    }
    return value;
}

bool LinkerImpl::map_builtin(const std::string &name, uint64_t &offset) {
    auto it = m_builtin_map.find(name);
    if (it == m_builtin_map.end()) {
        offset = 0;
        return false;
    }
    offset = it->second;
    return true;
}

void LinkerImpl::bind_symbol(const std::string &name, int local_index, int entry_index) {
    auto result = m_symbol_map.emplace(name, std::pair<int, int>{local_index, entry_index});
    if (!result.second) {
        throw std::runtime_error("Duplicate definition of symbol " + name);
    }
}

bool LinkerImpl::map_symbol(const std::string &name, int &local_index, int &entry_index) {
    auto it = m_symbol_map.find(name);
    if (it == m_symbol_map.end()) {
        local_index = -1;
        entry_index = -1;
        return false;
    }
    local_index = it->second.first;
    entry_index = it->second.second;
    return true;
}

void LinkerImpl::bind_symbol_section(int index, int local_index) {
    m_symbol_section_map.emplace(index, local_index);
}

bool LinkerImpl::map_symbol_section(int index, int &local_index) {
    auto it = m_symbol_section_map.find(index);
    if (it == m_symbol_section_map.end()) {
        local_index = -1;
        return false;
    }
    local_index = it->second;
    return true;
}

void LinkerImpl::bind_code_section(int index, int local_index) {
    m_code_section_map.emplace(index, local_index);
}

bool LinkerImpl::map_code_section(int index, int &local_index) {
    auto it = m_code_section_map.find(index);
    if (it == m_code_section_map.end()) {
        local_index = -1;
        return false;
    }
    local_index = it->second;
    return true;
}

void LinkerImpl::check_code_offset(uint64_t offset, uint64_t size) {
    if (offset + size > m_code_end) {
        std::string msg = "Code offset interval [" + std::to_string(offset) +
            ", " + std::to_string(offset + size) + ") is out of range";
        throw std::runtime_error(msg);
    }
}

void LinkerImpl::diag_reloc_entry(uint64_t offset, uint64_t value) {
    if (!DIAG_RELOC_ENABLED) {
        return;
    }
    printf("---- Reloc value at %zx: %zx (%d)\n", size_t(offset), size_t(value), int(value)); 
}

void LinkerImpl::diag_reloc_code(const char *tag, uint64_t offset) {
    if (!DIAG_RELOC_ENABLED) {
        return;
    }
    printf("%s %02x %02x %02x %02x\n", 
        tag, m_code[offset], m_code[offset+1], m_code[offset + 2], m_code[offset + 3]);
}

void LinkerImpl::diag_reloc_code_c(const char *tag, uint64_t offset) {
    if (!DIAG_RELOC_ENABLED) {
        return;
    }
    printf("%s %02x %02x\n", tag, m_code[offset], m_code[offset+1]);
}

} // linker
} // riscv

