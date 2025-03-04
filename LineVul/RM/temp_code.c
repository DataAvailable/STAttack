int kvm_arch_vcpu_ioctl_set_sregs (struct   kvm_vcpu *vcpu, struct   kvm_sregs *sregs) {
    struct   msr_data apic_base_msr;
    int mmu_reset_needed = 0;
    int pending_vec, max_bits, idx;
    struct   desc_ptr dt;
    if (!guest_cpuid_has_xsave (vcpu) && (sregs->cr4 & X86_CR4_OSXSAVE))
        return -EINVAL;
    dt.size = sregs->idt.limit;
    dt.address = sregs->idt.base;
    kvm_x86_ops->set_idt (vcpu, &dt);
    dt.size = sregs->gdt.limit;
    dt.address = sregs->gdt.base;
    kvm_x86_ops->set_gdt (vcpu, &dt);
    vcpu->arch.cr2 = sregs->cr2;
    mmu_reset_needed |= kvm_read_cr3 (vcpu) != sregs->cr3;
    vcpu->arch.cr3 = sregs->cr3;
    __set_bit (VCPU_EXREG_CR3, (ulong *) & vcpu -> arch.regs_avail);
    kvm_set_cr8 (vcpu, sregs->cr8);
    mmu_reset_needed |= vcpu->arch.efer != sregs->efer;
    kvm_x86_ops->set_efer (vcpu, sregs->efer);
    apic_base_msr.data = sregs->apic_base;
    apic_base_msr.host_initiated = true;
    kvm_set_apic_base (vcpu, &apic_base_msr);
    mmu_reset_needed |= kvm_read_cr0 (vcpu) != sregs->cr0;
    kvm_x86_ops->set_cr0 (vcpu, sregs->cr0);
    vcpu->arch.cr0 = sregs->cr0;
    mmu_reset_needed |= kvm_read_cr4 (vcpu) != sregs->cr4;
    kvm_x86_ops->set_cr4 (vcpu, sregs->cr4);
    if (sregs->cr4 & X86_CR4_OSXSAVE)
        kvm_update_cpuid (vcpu);
    idx = srcu_read_lock (&vcpu->kvm->srcu);
    if (!is_long_mode (vcpu) && is_pae (vcpu)) {
        load_pdptrs (vcpu, vcpu->arch.walk_mmu, kvm_read_cr3 (vcpu));
        mmu_reset_needed = 1;
    }
    srcu_read_unlock (&vcpu->kvm->srcu, idx);
    if (mmu_reset_needed)
        kvm_mmu_reset_context (vcpu);
    max_bits = KVM_NR_INTERRUPTS;
    pending_vec = find_first_bit ((const  unsigned  long  *) sregs->interrupt_bitmap, max_bits);
    if (pending_vec < max_bits) {
        kvm_queue_interrupt (vcpu, pending_vec, false);
        pr_debug ("Set back pending irq %d\n", pending_vec);
    }
    kvm_set_segment (vcpu, &sregs->cs, VCPU_SREG_CS);
    kvm_set_segment (vcpu, &sregs->ds, VCPU_SREG_DS);
    kvm_set_segment (vcpu, &sregs->es, VCPU_SREG_ES);
    kvm_set_segment (vcpu, &sregs->fs, VCPU_SREG_FS);
    kvm_set_segment (vcpu, &sregs->gs, VCPU_SREG_GS);
    kvm_set_segment (vcpu, &sregs->ss, VCPU_SREG_SS);
    kvm_set_segment (vcpu, &sregs->tr, VCPU_SREG_TR);
    kvm_set_segment (vcpu, &sregs->ldt, VCPU_SREG_LDTR);
    update_cr8_intercept (vcpu);
    if (kvm_vcpu_is_bsp (vcpu) && kvm_rip_read (vcpu) == 0xfff0 && sregs->cs.selector == 0xf000 && sregs->cs.base == 0xffff0000 && !is_protmode (vcpu))
        vcpu->arch.mp_state = KVM_MP_STATE_RUNNABLE;
    kvm_make_request (KVM_REQ_EVENT, vcpu);
    return 0;
}

