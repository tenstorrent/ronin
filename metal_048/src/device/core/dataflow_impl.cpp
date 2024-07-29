
#include <cstdint>

#include "core/kernel_structs.hpp"
#include "core/addr_map.hpp"
#include "core/sync.hpp"
#include "core/memory.hpp"
#include "core/cb_api.hpp"
#include "core/noc_api.hpp"
#include "core/dataflow_impl.hpp"

namespace tt {
namespace metal {
namespace device {

namespace {

uint32_t align(uint32_t addr, uint32_t alignment) { 
    return ((addr - 1) | (alignment - 1)) + 1; 
}

uint32_t mul_with_tile_size(DataFormat format, uint32_t index) {
    switch (DataFormat(uint32_t(format) & 0x1F)) {
    case DataFormat::Float16:
    case DataFormat::Float16_b: 
        return index << 11;
    case DataFormat::Bfp8_b:
    // Keep default as Bfp8?
    default: 
        return (index << 10) + (index << 6);
    }
}

} // namespace

//
//    DataflowImpl
//

DataflowImpl::DataflowImpl(
        Sync *sync,
        Memory *l1,
        CB *cb,
        NocArch *noc_arch,
        Noc *noc,
        uint32_t noc_index,
        uint32_t my_x,
        uint32_t my_y):
            m_sync(sync),
            m_l1(l1),
            m_cb(cb),
            m_noc_arch(noc_arch),
            m_noc(noc),
            m_noc_index(noc_index),
            m_my_x(my_x),
            m_my_y(my_y) { 
    m_num_dram_banks = m_noc_arch->num_dram_banks();
    m_num_l1_banks = m_noc_arch->num_l1_banks();
    m_noc_size_x = m_noc_arch->noc_size_x();
    m_noc_size_y = m_noc_arch->noc_size_y();
    m_pcie_core_noc_encoding = 
        m_noc_arch->noc_xy_encoding(m_noc_arch->pcie_noc_x(), m_noc_arch->pcie_noc_y());
    m_l1_arg_base = 
        (noc_index == 0) ? 
            AddrMap::BRISC_L1_ARG_BASE : 
            AddrMap::NCRISC_L1_ARG_BASE;
}

DataflowImpl::~DataflowImpl() { }

void DataflowImpl::reset() {
    // nothing to do
}

uint32_t DataflowImpl::get_arg_uint32(int arg_idx) {
    uint32_t *arg_base = reinterpret_cast<uint32_t *>(m_l1->map_addr(m_l1_arg_base));
    return arg_base[arg_idx];
}

void DataflowImpl::cb_push_back(uint32_t operand, uint32_t num_pages) {
    m_cb->cb_push_back(operand, num_pages);
}

void DataflowImpl::cb_pop_front(uint32_t operand, uint32_t num_pages) {
    m_cb->cb_pop_front(operand, num_pages);
}

int32_t DataflowImpl::get_tile_size(uint32_t operand) {
    return m_cb->get_tile_size(operand);
}

DataFormat DataflowImpl::get_dataformat(uint32_t operand) {
    return m_cb->get_unpack_src_format(operand);
}

uint32_t DataflowImpl::get_write_ptr(uint32_t operand) {
    return m_cb->get_write_ptr(operand);
}

uint32_t DataflowImpl::get_read_ptr(uint32_t operand) {
    return m_cb->get_read_ptr(operand);
}

void DataflowImpl::wait_for_sync_register_value(uint32_t addr, int32_t val) {
    volatile int32_t *reg_ptr = reinterpret_cast<volatile int32_t *>(m_l1->map_addr(addr));
    auto cond = [=]() -> bool {
        return (reg_ptr[0] == val);
    };
    m_sync->wait(cond);
}

void DataflowImpl::cb_reserve_back(uint32_t operand, uint32_t num_pages) {
    m_cb->cb_reserve_back(operand, num_pages);
}

void DataflowImpl::cb_wait_front(uint32_t operand, uint32_t num_pages) {
    m_cb->cb_wait_front(operand, num_pages);
}

uint64_t DataflowImpl::get_noc_multicast_addr(
        uint32_t noc_x_start,
        uint32_t noc_y_start,
        uint32_t noc_x_end,
        uint32_t noc_y_end,
        uint32_t addr) {
    return m_noc_arch->noc_multicast_addr(
        enc_noc_x(noc_x_start), 
        enc_noc_y(noc_y_start), 
        enc_noc_x(noc_x_end), 
        enc_noc_y(noc_y_end), 
        addr);
}

uint64_t DataflowImpl::get_noc_addr_remote(uint32_t noc_x, uint32_t noc_y, uint32_t addr) {
    return m_noc_arch->noc_xy_addr(enc_noc_x(noc_x), enc_noc_y(noc_y), addr);
}

uint64_t DataflowImpl::get_noc_addr_helper(uint32_t noc_xy, uint32_t addr) {
    // shift by 32 even if (NOC_ADDR_LOCAL_BITS > 32)
    // because 'noc_xy' contais XY value already shifted by (NOC_ADDR_LOCAL_BITS - 32)
    // (by construction of 'bank_to_noc_xy' tables)
    return (uint64_t(noc_xy) << 32) | addr;
}

uint64_t DataflowImpl::get_dram_noc_addr(
        uint32_t id, 
        uint32_t page_size, 
        uint32_t bank_base_address, 
        uint32_t offset) {
    uint32_t bank_id = id % m_num_dram_banks;
    uint32_t addr = (id / m_num_dram_banks) * align(page_size, 32) + bank_base_address + offset;
    addr += m_noc_arch->bank_to_dram_offset(bank_id);
    uint32_t noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t DataflowImpl::get_l1_noc_addr(
        uint32_t id, 
        uint32_t page_size, 
        uint32_t bank_base_address, 
        uint32_t offset) {
    uint32_t bank_id = id % m_num_l1_banks;
    uint32_t addr = (id / m_num_l1_banks) * align(page_size, 32) + bank_base_address + offset;
    addr += m_noc_arch->bank_to_l1_offset(bank_id);
    uint32_t noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t DataflowImpl::get_system_memory_noc_addr(
        uint32_t id, 
        uint32_t page_size, 
        uint32_t base_addr, 
        uint32_t offset) {
    // ACHTUNG: Apparent bug in original code (<< 32 is for Grayskull only)
    uint32_t addr = base_addr + page_size * id + offset;
    uint64_t noc_addr = m_noc_arch->noc_xy_addr2(m_pcie_core_noc_encoding, addr);
    return noc_addr;
}

uint64_t DataflowImpl::get_noc_addr_interleaved(
        uint32_t id, 
//        const InterleavedAddrGen<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        uint32_t offset) {
    // based on 'InterleavedAddrGen::get_noc_addr'
    uint32_t addr;
    uint32_t noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        addr = (id / m_num_dram_banks) * align(page_size, 32) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_dram_offset(bank_id);
        noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        addr = (id / m_num_l1_banks) * align(page_size, 32) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_l1_offset(bank_id);
        noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t DataflowImpl::get_noc_addr_interleaved_pow2(
        uint32_t id, 
//        const InterleavedPow2AddrGen<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t offset) {
    // based on 'InterleavedPow2AddrGen::get_noc_addr'
    // So far, only using this for DRAM,
    // but will eventually generalize to allow usage in L1 as well
    uint32_t addr;
    uint32_t noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        addr = ((id / m_num_dram_banks) << log_base_2_of_page_size) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_dram_offset(bank_id);
        noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        addr = ((id / m_num_l1_banks) << log_base_2_of_page_size) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_l1_offset(bank_id);
        noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

uint64_t DataflowImpl::get_noc_addr_interleaved_fast(
        uint32_t id, 
//        const InterleavedAddrGenFast<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        DataFormat data_format,
        uint32_t offset) {
    // based on 'InterleavedAddrGenFast::get_noc_addr'
    uint32_t addr;
    uint32_t noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        addr = mul_with_tile_size(data_format, id / m_num_dram_banks) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_dram_offset(bank_id);
        noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        addr = mul_with_tile_size(data_format, id / m_num_l1_banks) + bank_base_address + offset;
        addr += m_noc_arch->bank_to_l1_offset(bank_id);
        noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }
    uint64_t noc_addr = get_noc_addr_helper(noc_xy, addr);
    return noc_addr;
}

#if 0 // TODO: Revise this
uint64_t DataflowImpl::get_noc_addr_local(uint32_t addr) {
    return m_noc_arch->noc_xy_addr(m_my_x, m_my_y, addr);
}
#endif

uint64_t DataflowImpl::get_noc_addr_local(uint32_t addr) {
    uint32_t x = enc_noc_x(m_my_x);
    uint32_t y = enc_noc_y(m_my_y);
    return m_noc_arch->noc_xy_addr(x, y, addr);
}

void DataflowImpl::noc_async_read(
        uint64_t src_noc_addr, 
        uint32_t dst_local_l1_addr, 
        uint32_t size) {
    m_noc->fast_read_any_len(
        m_noc_index, 
        Noc::RD_CMD_BUF, 
        src_noc_addr, 
        dst_local_l1_addr, 
        size);
}

void DataflowImpl::noc_async_read_one_packet(
        uint64_t src_noc_addr, 
        uint32_t dst_local_l1_addr, 
        uint32_t size) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dst_local_l1_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, uint32_t(src_noc_addr));
    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, uint32_t(src_noc_addr >> 32));
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->incr_reads_num_issued(m_noc_index, 1);
}

void DataflowImpl::noc_async_read_one_packet_set_state(uint64_t src_noc_addr, uint32_t size) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, uint32_t(src_noc_addr >> 32));
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, size);
}

void DataflowImpl::noc_async_read_one_packet_with_state(
        uint32_t src_noc_addr, 
        uint32_t dst_local_l1_addr,
        bool inc_num_issued) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dst_local_l1_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_noc_addr);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    if (inc_num_issued) {
        m_noc->incr_reads_num_issued(m_noc_index, 1);
    }
}

void DataflowImpl::noc_async_read_set_state(uint64_t src_noc_addr) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, uint32_t(src_noc_addr >> 32));
}

void DataflowImpl::noc_async_read_with_state(
        uint32_t src_noc_addr, 
        uint32_t dst_local_l1_addr, 
        uint32_t size,
        bool inc_num_issued) {
    uint32_t max_burst_size = m_noc->max_burst_size();
    while (size > max_burst_size) {
        m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

        m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dst_local_l1_addr);
        m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_noc_addr);
        m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, max_burst_size);
        m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

        size -= max_burst_size;
        src_noc_addr += max_burst_size;
        dst_local_l1_addr += max_burst_size;
        if (inc_num_issued) {
            m_noc->incr_reads_num_issued(m_noc_index, 1);
        }
    }

    // left-over packet
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dst_local_l1_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_noc_addr);
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    if (inc_num_issued) {
        m_noc->incr_reads_num_issued(m_noc_index, 1);
    }
}

void DataflowImpl::noc_async_read_inc_num_issued(uint32_t num_issued_reads_inc) {
    m_noc->incr_reads_num_issued(m_noc_index, num_issued_reads_inc);
}

void DataflowImpl::noc_async_write_one_packet(
        uint32_t src_local_l1_addr, 
        uint64_t dst_noc_addr, 
        uint32_t size) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->write_ctrl_cpy_wr(
        m_noc_index, 
        Noc::WR_REG_CMD_BUF,
        Noc::UNICAST_WRITE_VC,
        false,
        false,
        false,
        true);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, src_local_l1_addr);
    m_noc->write_ret_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, uint32_t(dst_noc_addr));
    m_noc->write_ret_addr_mid(m_noc_index, Noc::WR_REG_CMD_BUF, uint32_t(dst_noc_addr >> 32));
    m_noc->write_at_len_be(m_noc_index, Noc::WR_REG_CMD_BUF, size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->incr_nonposted_writes_num_issued(m_noc_index, 1);
    m_noc->incr_nonposted_writes_acked(m_noc_index, 1);
}

void DataflowImpl::noc_async_write_one_packet_set_state(
        uint64_t dst_noc_addr, 
        uint32_t size,
        bool non_posted) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->write_ctrl_cpy_wr(
        m_noc_index, 
        Noc::WR_REG_CMD_BUF,
        Noc::UNICAST_WRITE_VC,
        false,
        false,
        false,
        non_posted);
    m_noc->write_ret_addr_mid(m_noc_index, Noc::WR_REG_CMD_BUF, uint32_t(dst_noc_addr >> 32));
    m_noc->write_at_len_be(m_noc_index, Noc::WR_REG_CMD_BUF, size);
}

void DataflowImpl::noc_async_write_one_packet_with_state(
        uint32_t src_local_l1_addr, 
        uint32_t dst_noc_addr,
        bool non_posted) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->write_targ_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, src_local_l1_addr);
    m_noc->write_ret_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, dst_noc_addr);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::WR_REG_CMD_BUF);

    if (non_posted) {
        m_noc->incr_nonposted_writes_num_issued(m_noc_index, 1);
        m_noc->incr_nonposted_writes_acked(m_noc_index, 1); // num_dests
    }
}

void DataflowImpl::noc_async_read_tile(
        uint32_t id, 
//        const InterleavedAddrGenFast<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        DataFormat data_format,
        uint32_t dst_local_l1_addr, 
        uint32_t offset) {
    // based on 'InterleavedAddrGenFast::noc_async_read_tile'
    uint32_t src_addr;
    uint32_t src_noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        src_addr = 
            mul_with_tile_size(data_format, id / m_num_dram_banks) + 
                bank_base_address + offset;
        src_addr += m_noc_arch->bank_to_dram_offset(bank_id);
        src_noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        src_addr = 
            mul_with_tile_size(data_format, id / m_num_l1_banks) + 
                bank_base_address + offset;
        src_addr += m_noc_arch->bank_to_l1_offset(bank_id);
        src_noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }

    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dst_local_l1_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_addr);
    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, src_noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, page_size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->incr_reads_num_issued(m_noc_index, 1);
}

void DataflowImpl::noc_async_read_page(
        uint32_t id, 
//        const InterleavedPow2AddrGenFast &s,
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t dest_addr, 
        uint32_t offset) {
    // based on 'InterleavedPow2AddrGenFast::noc_async_read_page'
    uint32_t src_addr;
    uint32_t src_noc_xy;

    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        src_addr = 
            ((id / m_num_dram_banks) << log_base_2_of_page_size) + 
                bank_base_address + offset;
        src_addr += m_noc_arch->bank_to_dram_offset(bank_id);
        src_noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        src_addr = 
            ((id / m_num_l1_banks) << log_base_2_of_page_size) + 
                bank_base_address + offset;
        src_addr += m_noc_arch->bank_to_l1_offset(bank_id);
        src_noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }

    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dest_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_addr);
    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, src_noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, 1 << log_base_2_of_page_size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->incr_reads_num_issued(m_noc_index, 1);
}

void DataflowImpl::noc_async_read_partial_page(
        uint32_t id, 
//        const InterleavedPow2AddrGenFast &s,
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t dest_addr, 
        uint32_t size, 
        uint32_t offset) {
    // based on 'InterleavedPow2AddrGenFast::noc_async_read_partial_page'
    uint32_t src_addr;
    uint32_t src_noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        src_addr = 
            ((id / m_num_dram_banks) << log_base_2_of_page_size) + 
                bank_base_address + offset;
        src_addr += m_noc_arch->bank_to_dram_offset(bank_id);
        src_noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        src_addr = 
            ((id / m_num_l1_banks) << log_base_2_of_page_size) + 
                bank_base_address + offset;
        src_addr += m_noc_arch->bank_to_l1_offset(bank_id);
        src_noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }

    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dest_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_addr);
    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, src_noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->incr_reads_num_issued(m_noc_index, 1);
}

void DataflowImpl::noc_async_write(
        uint32_t src_local_l1_addr, 
        uint64_t dst_noc_addr, 
        uint32_t size) {
    m_noc->fast_write_any_len(
        m_noc_index,
        Noc::WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr,
        size,
        Noc::UNICAST_WRITE_VC,
        false,
        false,
        1);
}

void DataflowImpl::noc_async_write_tile(
        uint32_t id, 
//        const InterleavedAddrGenFast<DRAM> &s, 
        bool dram,
        uint32_t bank_base_address,
        uint32_t page_size,
        DataFormat data_format,
        uint32_t src_local_l1_addr) {
    // based on 'InterleavedAddrGenFast::noc_async_write_tile'
    uint32_t dest_addr;
    uint32_t dest_noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        dest_addr = mul_with_tile_size(data_format, id / m_num_dram_banks) + bank_base_address;
        dest_addr += m_noc_arch->bank_to_dram_offset(bank_id);
        dest_noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        dest_addr = mul_with_tile_size(data_format, id / m_num_l1_banks) + bank_base_address;
        dest_addr += m_noc_arch->bank_to_l1_offset(bank_id);
        dest_noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }

    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->write_ctrl_cpy_wr(
        m_noc_index, 
        Noc::WR_REG_CMD_BUF,
        Noc::UNICAST_WRITE_VC,
        false, 
        false,
        false,
        true);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, src_local_l1_addr);
    m_noc->write_ret_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, dest_addr);
    m_noc->write_ret_addr_mid(m_noc_index, Noc::WR_REG_CMD_BUF, dest_noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::WR_REG_CMD_BUF, page_size);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->incr_nonposted_writes_num_issued(m_noc_index, 1);
    m_noc->incr_nonposted_writes_acked(m_noc_index, 1); // num_dests
}

void DataflowImpl::noc_async_write_page(
        uint32_t id, 
//        const InterleavedPow2AddrGenFast &s,
        bool dram,
        uint32_t bank_base_address,
        uint32_t log_base_2_of_page_size,
        uint32_t src_addr, 
        uint32_t write_size_bytes, 
        uint32_t offset) {
    // based on 'InterleavedPow2AddrGenFast::noc_async_write_page'
    uint32_t dest_addr;
    uint32_t dest_noc_xy;
    if (dram) {
        uint32_t bank_id = id % m_num_dram_banks;
        dest_addr = 
            ((id / m_num_dram_banks) << log_base_2_of_page_size) + 
                bank_base_address + offset;
        dest_addr += m_noc_arch->bank_to_dram_offset(bank_id);
        dest_noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    } else {
        uint32_t bank_id = id % m_num_l1_banks;
        dest_addr = 
            ((id / m_num_l1_banks) << log_base_2_of_page_size) + 
                bank_base_address + offset;
        dest_addr += m_noc_arch->bank_to_l1_offset(bank_id);
        dest_noc_xy = m_noc_arch->l1_bank_to_noc_xy(m_noc_index, bank_id);
    }

    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->write_ctrl_cpy_wr(
        m_noc_index, 
        Noc::WR_REG_CMD_BUF,
        Noc::UNICAST_WRITE_VC,
        false,
        false,
        false,
        true);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, src_addr);
    m_noc->write_ret_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, dest_addr);
    m_noc->write_ret_addr_mid(m_noc_index, Noc::WR_REG_CMD_BUF, dest_noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::WR_REG_CMD_BUF, write_size_bytes);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->incr_nonposted_writes_num_issued(m_noc_index, 1);
    m_noc->incr_nonposted_writes_acked(m_noc_index, 1); // num_dests
}

uint32_t DataflowImpl::get_semaphore(uint32_t semaphore_id) {
    return AddrMap::SEMAPHORE_BASE + semaphore_id * sizeof(uint32_t);
}

void DataflowImpl::noc_semaphore_set_remote(uint32_t src_local_l1_addr, uint64_t dst_noc_addr) {
    m_noc->fast_write_any_len(
        m_noc_index,
        Noc::WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr,
        4, // size in bytes
        Noc::UNICAST_WRITE_VC,
        false,
        false,
        1);
}

void DataflowImpl::noc_async_write_multicast(
        uint32_t src_local_l1_addr,
        uint64_t dst_noc_addr_multicast,
        uint32_t size,
        uint32_t num_dests) {
    m_noc->fast_write_any_len(
        m_noc_index,
        Noc::WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        Noc::MULTICAST_WRITE_VC,
        true,
        false,
        num_dests);
}

void DataflowImpl::noc_semaphore_set_multicast(
        uint32_t src_local_l1_addr, 
        uint64_t dst_noc_addr_multicast, 
        uint32_t num_dests) {
    m_noc->fast_write_any_len(
        m_noc_index,
        Noc::WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        4, //size in bytes
        Noc::MULTICAST_WRITE_VC,
        true,
        false,
        num_dests);
}

void DataflowImpl::noc_async_write_multicast_loopback_src(
        uint32_t src_local_l1_addr,
        uint64_t dst_noc_addr_multicast,
        uint32_t size,
        uint32_t num_dests) {
    m_noc->fast_write_any_len_loopback_src(
        m_noc_index,
        Noc::WR_REG_CMD_BUF,
        src_local_l1_addr,
        dst_noc_addr_multicast,
        size,
        Noc::MULTICAST_WRITE_VC,
        true,
        false,
        num_dests);
}

void DataflowImpl::noc_async_read_barrier() {
    m_noc->wait_reads_flushed(m_noc_index);
}

void DataflowImpl::noc_async_write_barrier() {
    m_noc->wait_nonposted_writes_flushed(m_noc_index);
}

void DataflowImpl::noc_semaphore_wait(volatile uint32_t *sem_addr, uint32_t val) {
    auto cond = [=]() -> bool {
        return (*sem_addr == val);
    };
    m_sync->wait(cond);
}

void DataflowImpl::noc_semaphore_set(volatile uint32_t *sem_addr, uint32_t val) {
    *sem_addr = val;
}

void DataflowImpl::noc_semaphore_inc(uint64_t addr, uint32_t incr) {
    m_noc->fast_atomic_increment(
        m_noc_index, 
        Noc::AT_CMD_BUF, 
        addr, 
        incr, 
        31,     // wrap
        false); // linked
}

void DataflowImpl::noc_fast_read(uint32_t src_addr, uint32_t dest_addr) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dest_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, src_addr);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);
}

void DataflowImpl::noc_fast_read_set_src_xy(uint64_t src_addr) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, uint32_t(src_addr >> 32));
}

void DataflowImpl::noc_fast_read_set_len(uint32_t len_bytes) {
    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, len_bytes); 
}

void DataflowImpl::noc_fast_read_inc_num_issued(uint32_t num_issued) {
    m_noc->incr_reads_num_issued(m_noc_index, num_issued);
}

void DataflowImpl::noc_fast_write(uint32_t src_addr, uint64_t dest_addr) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_CMD_BUF);

    m_noc->write_targ_addr_lo(m_noc_index, Noc::WR_CMD_BUF, src_addr);
    m_noc->write_ret_addr_lo(m_noc_index, Noc::WR_CMD_BUF, uint32_t(dest_addr));
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::WR_CMD_BUF);
}

void DataflowImpl::noc_fast_write_set_cmd_field(uint32_t vc, bool mcast, bool linked) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_CMD_BUF);

    m_noc->write_ctrl_cpy_wr(m_noc_index, Noc::WR_CMD_BUF, vc, mcast, linked, false, true);
}

void DataflowImpl::noc_fast_write_set_dst_xy(uint64_t dest_addr) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_CMD_BUF);

    m_noc->write_ret_addr_mid(m_noc_index, Noc::WR_CMD_BUF, uint32_t(dest_addr >> 32));
}

void DataflowImpl::noc_fast_write_set_len(uint32_t len_bytes) {
    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_CMD_BUF);

    m_noc->write_at_len_be(m_noc_index, Noc::WR_CMD_BUF, len_bytes);
}

void DataflowImpl::noc_fast_write_inc_num_dests(uint32_t num_issued) {
    m_noc->incr_nonposted_writes_num_issued(m_noc_index, num_issued);
    m_noc->incr_nonposted_writes_acked(m_noc_index, num_issued);
}

void DataflowImpl::cq_wait_front() {
    // TODO
}

void DataflowImpl::notify_host_of_cq_read_pointer() {
    // TODO
}

void DataflowImpl::cq_pop_front(uint32_t cmd_size_B) {
    // TODO
}

void DataflowImpl::noc_async_read_global_dram(
        uint32_t dst_addr,
        uint32_t src_addr,
        uint32_t src_log2_page_size,
        uint32_t src_offset,
        uint32_t len_bytes) {
    uint64_t noc_addr = get_noc_addr_global_dram(src_addr, src_log2_page_size, src_offset);
    noc_fast_read_global(dst_addr, noc_addr, len_bytes);
}

void DataflowImpl::noc_async_read_global_l1(
        uint32_t dst_addr,
        uint32_t src_addr,
        uint32_t src_log2_page_size,
        uint32_t src_offset,
        uint32_t len_bytes) {
    uint64_t noc_addr = get_noc_addr_global_l1(src_addr, src_log2_page_size, src_offset);
    noc_fast_read_global(dst_addr, noc_addr, len_bytes);
}

void DataflowImpl::noc_async_write_global_dram(
        uint32_t src_addr,
        uint32_t dst_addr,
        uint32_t dst_log2_page_size,
        uint32_t dst_offset,
        uint32_t len_bytes) {
    uint64_t noc_addr = get_noc_addr_global_dram(dst_addr, dst_log2_page_size, dst_offset);
    noc_fast_write_global(src_addr, noc_addr, len_bytes);
}

void DataflowImpl::noc_async_write_global_l1(
        uint32_t src_addr,
        uint32_t dst_addr,
        uint32_t dst_log2_page_size,
        uint32_t dst_offset,
        uint32_t len_bytes) {
    uint64_t noc_addr = get_noc_addr_global_l1(dst_addr, dst_log2_page_size, dst_offset);
    noc_fast_write_global(src_addr, noc_addr, len_bytes);
}

uint64_t DataflowImpl::get_noc_addr_global_dram(
        uint32_t base_addr, 
        uint32_t log2_page_size, 
        uint32_t offset) {
    uint32_t id = offset >> log2_page_size;
    offset -= id << log2_page_size;
    uint32_t bank_id = id % m_num_dram_banks;
    uint32_t addr = ((id / m_num_dram_banks) << log2_page_size) + base_addr + offset;
    addr += m_noc_arch->bank_to_dram_offset(bank_id);
    uint32_t noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    return get_noc_addr_helper(noc_xy, addr);
}

uint64_t DataflowImpl::get_noc_addr_global_l1(
        uint32_t base_addr, 
        uint32_t log2_page_size, 
        uint32_t offset) {
    uint32_t id = offset >> log2_page_size;
    offset -= id << log2_page_size;
    uint32_t bank_id = id % m_num_l1_banks;
    uint32_t addr = ((id / m_num_l1_banks) << log2_page_size) + base_addr + offset;
    addr += m_noc_arch->bank_to_dram_offset(bank_id);
    uint32_t noc_xy = m_noc_arch->dram_bank_to_noc_xy(m_noc_index, bank_id);
    return get_noc_addr_helper(noc_xy, addr);
}

void DataflowImpl::noc_fast_read_global(
        uint32_t dst_addr, 
        uint64_t src_addr, 
        uint32_t len_bytes) {
    uint32_t addr = uint32_t(src_addr);
    uint32_t noc_xy = uint32_t(src_addr >> 32);

    m_noc->wait_fast_read_ok(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->write_ret_addr_lo(m_noc_index, Noc::RD_CMD_BUF, dst_addr);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::RD_CMD_BUF, addr);
    m_noc->write_targ_addr_mid(m_noc_index, Noc::RD_CMD_BUF, noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::RD_CMD_BUF, len_bytes);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::RD_CMD_BUF);

    m_noc->incr_reads_num_issued(m_noc_index, 1);
}

void DataflowImpl::noc_fast_write_global(
        uint32_t src_addr, 
        uint64_t dst_addr, 
        uint32_t len_bytes) {
    uint32_t addr = uint32_t(dst_addr);
    uint32_t noc_xy = uint32_t(dst_addr >> 32);

    m_noc->wait_fast_write_ok(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->write_ctrl_cpy_wr(
        m_noc_index, 
        Noc::WR_REG_CMD_BUF,
        Noc::UNICAST_WRITE_VC,
        false,
        false,
        false,
        true);
    m_noc->write_targ_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, src_addr);
    m_noc->write_ret_addr_lo(m_noc_index, Noc::WR_REG_CMD_BUF, addr);
    m_noc->write_ret_addr_mid(m_noc_index, Noc::WR_REG_CMD_BUF, noc_xy);
    m_noc->write_at_len_be(m_noc_index, Noc::WR_REG_CMD_BUF, len_bytes);
    m_noc->write_cmd_ctrl_send_req(m_noc_index, Noc::WR_REG_CMD_BUF);

    m_noc->incr_nonposted_writes_num_issued(m_noc_index, 1);
    m_noc->incr_nonposted_writes_acked(m_noc_index, 1);
}

uint32_t DataflowImpl::enc_noc_x(uint32_t x) {
    return (m_noc_index == 0) ? x : m_noc_size_x - 1 - x;
}

uint32_t DataflowImpl::enc_noc_y(uint32_t y) {
    return (m_noc_index == 0) ? y : m_noc_size_y - 1 - y;
}

} // namespace device
} // namespace metal
} // namespace tt

