f90_template = """module solve_ida

   implicit none

   integer :: neq = {neq}
   integer :: iout(50)
   real*8 :: rout(50)
   real*8 :: y0({neq}), yp0({neq})
   real*8 :: diff({neq}), mas({neq}, {neq})
   real*8 :: jac({neq}, {neq})
   real*8 :: rates({nrates})
   real*8 :: dypdr({neq}, {nrates})
   integer :: dvacdy({nvac}, {neq})

end module solve_ida

module ida_state

   ! SUNDIALS objects. Kept separate from solve_ida so that f2py does not
   ! need to know about them.
   use, intrinsic :: iso_c_binding
   use fsundials_core_mod
   use fida_mod
   use fsunmatrix_dense_mod

   implicit none

   type(c_ptr) :: ctx = c_null_ptr
   type(c_ptr) :: ida_mem = c_null_ptr
   type(N_Vector), pointer :: sv_y => null(), sv_yp => null()
   type(N_Vector), pointer :: sv_atol => null(), sv_id => null()
   type(N_Vector), pointer :: sv_constr => null()
   type(SUNMatrix), pointer :: sm_a => null()
   type(SUNLinearSolver), pointer :: sls => null()

contains

   ! IDA residual callback; wraps fidaresfun
   integer(c_int) function micki_resfn(t, sunvec_y, sunvec_yp, sunvec_r, &
                                       user_data) result(ierr) bind(C)

      real(c_double), value :: t
      type(N_Vector) :: sunvec_y, sunvec_yp, sunvec_r
      type(c_ptr), value :: user_data

      real(c_double), pointer :: y(:), yp(:), r(:)
      real*8 :: rpar(1)
      integer :: ipar(1), reserr

      y => FN_VGetArrayPointer(sunvec_y)
      yp => FN_VGetArrayPointer(sunvec_yp)
      r => FN_VGetArrayPointer(sunvec_r)

      call fidaresfun(t, y, yp, r, ipar, rpar, reserr)
      ierr = reserr

   end function micki_resfn

   ! IDA dense Jacobian callback; wraps fidadjac
   integer(c_int) function micki_jacfn(t, cj, sunvec_y, sunvec_yp, sunvec_r, &
                                       sunmat_j, user_data, tmp1, tmp2, tmp3) &
                                       result(ierr) bind(C)

      real(c_double), value :: t, cj
      type(N_Vector) :: sunvec_y, sunvec_yp, sunvec_r
      type(SUNMatrix) :: sunmat_j
      type(c_ptr), value :: user_data
      type(N_Vector) :: tmp1, tmp2, tmp3

      real(c_double), pointer :: y(:), yp(:), r(:), jac(:)
      real*8 :: rpar(1), ewt(1), wk(1)
      integer :: ipar(1), jacerr

      y => FN_VGetArrayPointer(sunvec_y)
      yp => FN_VGetArrayPointer(sunvec_yp)
      r => FN_VGetArrayPointer(sunvec_r)
      ! column-major neq x neq, matching fidadjac's jac(neqin, neqin)
      jac => FSUNDenseMatrix_Data(sunmat_j)

      call fidadjac(size(y), t, y, yp, r, jac, cj, ewt, 0.d0, ipar, rpar, &
                    wk, wk, wk, jacerr)
      ierr = jacerr

   end function micki_jacfn

   subroutine free_ida()

      integer(c_int) :: ier

      if (c_associated(ida_mem)) call FIDAFree(ida_mem)
      if (associated(sls)) ier = FSUNLinSolFree(sls)
      if (associated(sm_a)) call FSUNMatDestroy(sm_a)
      if (associated(sv_y)) call FN_VDestroy(sv_y)
      if (associated(sv_yp)) call FN_VDestroy(sv_yp)
      if (associated(sv_atol)) call FN_VDestroy(sv_atol)
      if (associated(sv_id)) call FN_VDestroy(sv_id)
      if (associated(sv_constr)) call FN_VDestroy(sv_constr)
      if (c_associated(ctx)) ier = FSUNContext_Free(ctx)
      ida_mem = c_null_ptr
      ctx = c_null_ptr
      nullify(sls, sm_a, sv_y, sv_yp, sv_atol, sv_id, sv_constr)

   end subroutine free_ida

end module ida_state

subroutine initialize(neqin, y0in, rtol, atol, ipar, rpar, id_vec, use_jac)

   use, intrinsic :: iso_c_binding
   use fsundials_core_mod
   use fida_mod
   use fnvector_serial_mod
   use fsunmatrix_dense_mod
   use fsunlinsol_lapackdense_mod
   use solve_ida, only: neq, iout, rout, y0, yp0, mas, diff, dypdr, dvacdy
   use ida_state

   implicit none

   integer, intent(in) :: neqin, ipar(*)
   real*8, intent(in) :: y0in(neqin), rtol, atol(*)
   real*8, intent(in) :: rpar(*)
   real*8, intent(in) :: id_vec(neqin)
   integer, intent(in) :: use_jac
   real*8 :: constr_vec(neqin)
   real*8 :: t0, yptmp(neqin)
   integer :: ier
   integer :: i
   integer(c_int32_t) :: n
   real(c_double), pointer :: v(:)

   dypdr = 0
{dypdrcalc}

   dvacdy = 0
{dvacdycalc}

   constr_vec = 1.d0

   y0 = y0in
   yp0 = 0
   yptmp = 0
   diff = id_vec
   mas = 0
   t0 = 0

   do i = 1, neq
      mas(i, i) = id_vec(i)
   enddo

   ! Calculate yp
   call fidaresfun(0.d0, y0, yptmp, yp0, ipar, rpar, ier)

   ! Release any solver left over from a previous call
   call free_ida()

   ! initialize Sundials
   ier = FSUNContext_Create(SUN_COMM_NULL, ctx)
   n = int(neq, c_int32_t)

   sv_y => FN_VNew_Serial(n, ctx)
   v => FN_VGetArrayPointer(sv_y)
   v = y0
   sv_yp => FN_VNew_Serial(n, ctx)
   v => FN_VGetArrayPointer(sv_yp)
   v = yp0
   sv_atol => FN_VNew_Serial(n, ctx)
   v => FN_VGetArrayPointer(sv_atol)
   v = atol(1:neq)
   sv_id => FN_VNew_Serial(n, ctx)
   v => FN_VGetArrayPointer(sv_id)
   v = id_vec
   sv_constr => FN_VNew_Serial(n, ctx)
   v => FN_VGetArrayPointer(sv_constr)
   v = constr_vec

   ! allocate memory
   ida_mem = FIDACreate(ctx)
   ier = FIDAInit(ida_mem, c_funloc(micki_resfn), t0, sv_y, sv_yp)
   ier = FIDASVtolerances(ida_mem, rtol, sv_atol)
   ! set maximum number of steps (default = 500)
   ier = FIDASetMaxNumSteps(ida_mem, 50000_c_long)
   ! set algebraic variables
   ier = FIDASetId(ida_mem, sv_id)
   ! set constraints (all yi >= 0.)
   ier = FIDASetConstraints(ida_mem, sv_constr)

   ! Dense LAPACK linear solver. If use_jac is nonzero, the analytic
   ! Jacobian (fidadjac) is used; otherwise IDA falls back to its internal
   ! difference-quotient Jacobian (the behavior of micki with Sundials 4.X).
   sm_a => FSUNDenseMatrix(n, n, ctx)
   sls => FSUNLinSol_LapackDense(sv_y, sm_a, ctx)
   ier = FIDASetLinearSolver(ida_mem, sls, sm_a)
   if (use_jac /= 0) then
      ier = FIDASetJacFn(ida_mem, c_funloc(micki_jacfn))
   end if

end subroutine initialize

subroutine find_steady_state(neqin, nrates, dt, maxiter, epsilon, t1, u1, du1, r1)

   use, intrinsic :: iso_c_binding
   use fida_mod
   use solve_ida, only: y0, yp0, iout, rout, rates, dypdr
   use ida_state, only: ida_mem

   implicit none

   integer, intent(in) :: neqin, nrates, maxiter
   real*8, intent(in) :: dt, epsilon

   real*8 :: rpar(1)
   integer :: ipar(1)

   real*8, intent(out) :: t1, u1(neqin), du1(neqin), r1(nrates)

   real*8 :: tout, epsilon2
   real*8 :: dutmp(neqin), du0(neqin)
   integer :: ier
   integer :: i

   logical :: converged

   converged = .FALSE.
   epsilon2 = epsilon**2
   i = 0
   tout = 0.0d0
   u1 = y0
   du1 = yp0
   t1 = 0.d0
   du0 = 0.d0

   ier = FIDACalcIC(ida_mem, IDA_YA_YDP_INIT, dt)

   do while (.not. converged)
      if (tout - t1 < dt * 0.01) then
         tout = tout + dt
      end if

      call ida_step(tout, t1, u1, du1, ier)

      i = i + 1

      call fidaresfun(tout, u1, du0, dutmp, ipar, rpar, ier)

      if (maxval(dutmp**2) < epsilon2) then
         converged = .TRUE.
      end if
      if (i >= maxiter) then
         print *, "ODE NOT CONVERGED!"
         exit
      end if
   end do
   
   call ratecalc({neq}, u1)
   r1 = rates

end subroutine find_steady_state

subroutine solve(neqin, nrates, nt, tfinal, t1, u1, du1, r1)

   use solve_ida, only: y0, yp0, iout, rout, rates

   implicit none

   integer, intent(in) :: neqin, nt, nrates
   real*8, intent(in) :: tfinal

   real*8 :: rpar(1)
   integer :: ipar(1)

   real*8, intent(out) :: t1(nt)
   real*8, intent(out) :: u1(neqin, nt), du1(neqin, nt)
   real*8, intent(out) :: r1(nrates, nt)

   real*8 :: dt, tout
   integer :: ier
   integer :: i

   dt = tfinal / (nt - 1)
   tout = 0.0d0
   t1 = 0
   u1 = 0
   du1 = 0
   u1(:, 1) = y0
   du1(:, 1) = yp0
   t1(1) = 0.d0
   call ratecalc({neq}, u1(:, 1))
   r1(:, 1) = rates


!   ier = FIDACalcIC(ida_mem, IDA_YA_YDP_INIT, dt)

   do i = 2, nt
      tout = tout + dt
      do while (tout - t1(i) > dt * 0.01)
         call ida_step(tout, t1(i), u1(:, i), du1(:, i), ier)
      end do
      r1(:, i) = rates
   end do

end subroutine solve

subroutine finalize

   use ida_state, only: free_ida

   implicit none

   call free_ida()

end subroutine finalize

subroutine calc_res(neqin, y, yp, res, ier)

   ! Evaluate the DAE residual (for testing/debugging).

   implicit none

   integer, intent(in) :: neqin
   real*8, intent(in) :: y(neqin), yp(neqin)
   real*8, intent(out) :: res(neqin)
   integer, intent(out) :: ier

   real*8 :: rpar(1)
   integer :: ipar(1)

   call fidaresfun(0.d0, y, yp, res, ipar, rpar, ier)

end subroutine calc_res

subroutine calc_jac(neqin, y, yp, cj, jac, ier)

   ! Evaluate the analytic Jacobian dF/dy + cj dF/dyp (for testing/debugging).

   implicit none

   integer, intent(in) :: neqin
   real*8, intent(in) :: y(neqin), yp(neqin), cj
   real*8, intent(out) :: jac(neqin, neqin)
   integer, intent(out) :: ier

   real*8 :: r(neqin), rpar(1), ewt(1), wk(1)
   integer :: ipar(1)

   r = 0
   call fidadjac(neqin, 0.d0, y, yp, r, jac, cj, ewt, 0.d0, ipar, rpar, &
                 wk, wk, wk, ier)

end subroutine calc_jac

subroutine ida_step(tout, tret, u, du, ier)

   ! Advance IDA to tout (IDA_NORMAL) and copy the solution into u, du.

   use, intrinsic :: iso_c_binding
   use fsundials_core_mod
   use fida_mod
   use solve_ida, only: neq
   use ida_state, only: ida_mem, sv_y, sv_yp

   implicit none

   real*8, intent(in) :: tout
   real*8, intent(out) :: tret, u(neq), du(neq)
   integer, intent(out) :: ier

   real(c_double) :: tr(1)
   real(c_double), pointer :: v(:)

   ier = FIDASolve(ida_mem, tout, tr, sv_y, sv_yp, IDA_NORMAL)
   tret = tr(1)
   v => FN_VGetArrayPointer(sv_y)
   u = v
   v => FN_VGetArrayPointer(sv_yp)
   du = v

end subroutine ida_step

subroutine fidaresfun(tres, yin, ypin, res, ipar, rpar, reserr)

   use solve_ida, only: neq, diff, dypdr, rates

   implicit none

   integer, intent(in) :: ipar(*)
   integer, intent(out) :: reserr
   real*8, intent(in) :: tres, rpar(*)
   real*8, intent(in) :: yin(neq), ypin(neq)
   real*8, intent(out) :: res(neq)
   real*8 :: y(neq)

   integer :: i

   reserr = 0

   y = yin
   res = 0


   do i = 1, neq
      if (y(i) < -1d-10)  then
!         y(i) = 0.d0
         reserr = 1
      endif
   enddo

   call ratecalc({neq}, y)

   res = matmul(dypdr, rates) - diff * ypin
   
end subroutine fidaresfun

subroutine fidadjac(neqin, t, yin, ypin, r, jac, cj, ewt, h, ipar, rpar, wk1, wk2, wk3, djacerr)

   use solve_ida, only: mas, dypdr, dvacdy
    
   implicit none
   
   integer :: neqin, ipar(*)
   integer :: djacerr
   real*8 :: t, h, cj, rpar(*)
   real*8 :: yin(neqin), ypin(neqin), r(neqin), ewt(*), jac(neqin, neqin)
   real*8 :: wk1(*), wk2(*), wk3(*)
   real*8 :: y(neqin), drdy({nrates}, {neq}), drdvac({nrates}, {nvac}), vac({nvac})

   integer :: i

   djacerr = 0

   jac = 0
   y = yin

   do i = 1, neqin
      if (y(i) < -1d-10) then
         y(i) = 0.d0
         djacerr = 1
      endif
   enddo

   vac = 0
{vaccalc}

   do i = 1, {nvac}
      if (vac(i) < -1d-10) then
         vac(i) = 0.d0
         djacerr = 1
      endif
   enddo

   drdy = 0
{drdycalc}

   drdvac = 0
{drdvaccalc}

   drdy = drdy + matmul(drdvac, dvacdy)

   jac = matmul(dypdr, drdy) - cj * mas

end subroutine fidadjac

subroutine ratecalc(neqin, yin)

   use solve_ida, only: rates

   implicit none

   integer, intent(in) :: neqin
   real*8, intent(in) :: yin(neqin)
   real*8 :: y(neqin)
   real*8 :: vac({nvac})

   integer :: i

   y = yin


   vac = 0
{vaccalc}

   do i = 1, {nvac}
      if (vac(i) < -1d-10) then
         vac(i) = 0.d0
      endif
   enddo

   rates = 0
{ratecalc}

end subroutine ratecalc


   """


pyf_template = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.

python module {modname} ! in
    interface  ! in :{modname}
        module solve_ida ! in :{modname}:{modname}.f90
            integer dimension(50) :: iout
            real*8 dimension(50) :: rout
            real*8 dimension({neq}) :: y0
            real*8 dimension({neq}) :: yp0
            real*8 dimension({neq}) :: diff
            real*8 dimension({neq},{neq}) :: mas
            real*8 dimension({neq},{neq}) :: jac
            real*8 dimension({nrates}) :: rates
            real*8 dimension({neq},{nrates}) :: dypdr
            integer dimension({nvac},{neq}) :: dvacdy
            integer, optional :: neq={neq}
        end module solve_ida
        subroutine initialize(neqin,y0in,rtol,atol,ipar,rpar,id_vec,use_jac) ! in :{modname}:{modname}.f90
            use solve_ida, only: neq,iout,rout,y0,yp0,mas,diff,dypdr,dvacdy
            integer, optional,intent(in),check(len(y0in)>=neqin),depend(y0in) :: neqin=len(y0in)
            real*8 dimension(neqin),intent(in) :: y0in
            real*8 intent(in) :: rtol
            real*8 dimension(*),intent(in) :: atol
            integer dimension(*),intent(in) :: ipar
            real*8 dimension(*),intent(in) :: rpar
            real*8 dimension(neqin),intent(in),depend(neqin) :: id_vec
            integer intent(in) :: use_jac
        end subroutine initialize
        subroutine find_steady_state(neqin,nrates,dt,maxiter,epsilon,t1,u1,du1,r1) ! in :{modname}:{modname}.f90
            use solve_ida, only: y0,yp0,iout,rout,rates,dypdr
            integer intent(in) :: neqin
            integer intent(in) :: nrates
            real*8 intent(in) :: dt
            integer intent(in) :: maxiter
            real*8 intent(in) :: epsilon
            real*8 intent(out) :: t1
            real*8 intent(out),dimension(neqin),depend(neqin) :: u1
            real*8 intent(out),dimension(neqin),depend(neqin) :: du1
            real*8 intent(out),dimension(nrates),depend(nrates) :: r1
        end subroutine find_steady_state
        subroutine solve(neqin,nrates,nt,tfinal,t1,u1,du1,r1) ! in :{modname}:{modname}.f90
            use solve_ida, only: y0,yp0,iout,rout,rates
            integer intent(in) :: neqin
            integer intent(in) :: nrates
            integer intent(in) :: nt
            real*8 intent(in) :: tfinal
            real*8 intent(out),dimension(nt),depend(nt) :: t1
            real*8 intent(out),dimension(neqin,nt),depend(neqin,nt) :: u1
            real*8 intent(out),dimension(neqin,nt),depend(neqin,nt) :: du1
            real*8 intent(out),dimension(nrates,nt),depend(nrates,nt) :: r1
        end subroutine solve
        subroutine calc_res(neqin,y,yp,res,ier) ! in :{modname}:{modname}.f90
            integer, optional,intent(in),check(len(y)>=neqin),depend(y) :: neqin=len(y)
            real*8 dimension(neqin),intent(in) :: y
            real*8 dimension(neqin),intent(in),depend(neqin) :: yp
            real*8 dimension(neqin),intent(out),depend(neqin) :: res
            integer intent(out) :: ier
        end subroutine calc_res
        subroutine calc_jac(neqin,y,yp,cj,jac,ier) ! in :{modname}:{modname}.f90
            integer, optional,intent(in),check(len(y)>=neqin),depend(y) :: neqin=len(y)
            real*8 dimension(neqin),intent(in) :: y
            real*8 dimension(neqin),intent(in),depend(neqin) :: yp
            real*8 intent(in) :: cj
            real*8 dimension(neqin,neqin),intent(out),depend(neqin) :: jac
            integer intent(out) :: ier
        end subroutine calc_jac
        subroutine finalize ! in :{modname}:{modname}.f90
        end subroutine finalize
    end interface
end python module {modname}

! This file was auto-generated with f2py (version:2).
! See http://cens.ioc.ee/projects/f2py2e/"""
