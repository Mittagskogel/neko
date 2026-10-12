! Taylor-Green vortex (TGV)
!
! Time-integration of the Taylor-Green vortex up to specified time, evaluating
! total kinetic energy and enstrophy. The resulting flow fields may be
! visualised using Paraview or VisIt by opening the field0.nek5000 file.
!
module user
  use neko
  implicit none

  ! Global user variables
  type(field_t) :: w1

contains

  ! Register user-defined functions (see user_intf.f90)
  subroutine user_setup(user)
    type(user_t), intent(inout) :: user
    user%initial_conditions => initial_conditions
    user%mesh_setup => user_mesh_scale
    user%compute => user_calc_quantities
    user%initialize => user_initialize
    user%finalize => user_finalize
  end subroutine user_setup

  ! Rescale mesh
  subroutine user_mesh_scale(msh, time)
    type(mesh_t), intent(inout) :: msh
    type(time_state_t), intent(in) :: time
    integer :: i, p, nvert
    real(kind=rp) :: d
    d = 4._rp

    ! The original mesh has size 0..8 to be mapped onto -pi..pi
    ! will be updated later to a method giving back the vertices of the mesh
    nvert = size(msh%points)
    do i = 1, nvert
       msh%points(i)%x(1) = (msh%points(i)%x(1) - d) / d * pi
       msh%points(i)%x(2) = (msh%points(i)%x(2) - d) / d * pi
       msh%points(i)%x(3) = (msh%points(i)%x(3) - d) / d * pi
    end do

  end subroutine user_mesh_scale

  ! User-defined initial condition
  subroutine initial_conditions(scheme_name, fields)
    character(len=*), intent(in) :: scheme_name
    type(field_list_t), intent(inout) :: fields
    integer :: i, ntot
    real(kind=rp) :: uvw(3)
    type(dofmap_t), pointer :: dof
    type (field_t), pointer :: u, v, w, p

    dof => fields%dof(1)
    u => fields%get_by_name("u")
    v => fields%get_by_name("v")
    w => fields%get_by_name("w")
    p => fields%get_by_name("p")

    ntot = dof%size()
    do i = 1, ntot
       uvw = tgv_ic(u%dof%x(i,1,1,1),u%dof%y(i,1,1,1),u%dof%z(i,1,1,1))
       u%x(i,1,1,1) = uvw(1)
       v%x(i,1,1,1) = uvw(2)
       w%x(i,1,1,1) = uvw(3)
    end do

    call field_rzero(p)
  end subroutine initial_conditions

  function tgv_ic(x, y, z) result(uvw)
    real(kind=rp) :: x, y, z
    real(kind=rp) :: ux, uy, uz
    real(kind=rp) :: uvw(3)

    uvw(1) = sin(x)*cos(y)*cos(z)
    uvw(2) = -cos(x)*sin(y)*cos(z)
    uvw(3) = 0._rp
  end function tgv_ic

  ! User-defined initialization called just before time loop starts
  subroutine user_initialize(time)
    type(time_state_t), intent(in) :: time
    real(kind=rp) :: t
    type(field_t), pointer :: u

    u => neko_registry%get_field('u')

    ! initialize work arrays for postprocessing
    call w1%init(u%dof, 'work1')

    ! call usercheck and vorticity simcomp also for tstep=0
    call neko_simcomps%simcomps(1)%simcomp%compute(time)
    call user_calc_quantities(time)

  end subroutine user_initialize

  ! User-defined routine called at the end of every time step
  subroutine user_calc_quantities(time)
    type(time_state_t), intent(in) :: time

    type(field_t), pointer :: omega_x, omega_y, omega_z, u, v, w
    integer :: ntot, i
    real(kind=rp) :: vv, sum_e1(1), e1, e2, sum_e2(1), oo, e3
    type(coef_t), pointer :: coef

    if (mod(time%tstep, 50) .ne. 0) return

    coef => neko_user_access%case%fluid%c_Xh
    u => neko_registry%get_field('u')
    v => neko_registry%get_field('v')
    w => neko_registry%get_field('w')

    omega_x => neko_registry%get_field("omega_x")
    omega_y => neko_registry%get_field("omega_y")
    omega_z => neko_registry%get_field("omega_z")

!    Option 1:
!    sum_e1 = 0._rp
!    sum_e2 = 0._rp
!    do i = 1, ntot
!       vv = u%x(i,1,1,1)**2 + v%x(i,1,1,1)**2 + w%x(i,1,1,1)**2
!       oo = om1%x(i,1,1,1)**2 + om2%x(i,1,1,1)**2 + om3%x(i,1,1,1)**2
!       sum_e1 = sum_e1 + vv*coef%B(i,1,1,1)
!       sum_e2 = sum_e2 + oo*coef%B(i,1,1,1)
!    end do
!    e1 = 0.5 * glsum(sum_e1,1) / coef%volume
!    e2 = 0.5 * glsum(sum_e2,1) / coef%volume

!    Option 2:
!    do i = 1, ntot
!       w1%x(i,1,1,1) = u%x(i,1,1,1)**2 + v%x(i,1,1,1)**2 + w%x(i,1,1,1)**2
!       w2%x(i,1,1,1) = om1%x(i,1,1,1)**2 + om2%x(i,1,1,1)**2 + om3%x(i,1,1,1)**2
!    end do
!    e1 = 0.5 * glsc2(w1%x,coef%B,ntot) / coef%volume
!    e2 = 0.5 * glsc2(w2%x,coef%B,ntot) / coef%volume

!    Option 3:
!    w1%x = u%x**2 + v%x**2 + w%x**2
!    w2%x = om1%x**2 + om2%x**2 + om3%x**2
!    e1 = 0.5 * glsc2(w1%x,coef%B,ntot) / coef%volume
!    e2 = 0.5 * glsc2(w2%x,coef%B,ntot) / coef%volume

!    Option 4:

    call field_col3(w1, u, u)
    call field_addcol3(w1, v, v)
    call field_addcol3(w1, w, w)
    if (NEKO_BCKND_DEVICE .eq. 1) then
       e1 = 0.5_rp * device_glsc2(w1%x_d, coef%B_d, w1%size()) / coef%volume
    else
       e1 = 0.5_rp * glsc2(w1%x, coef%B, w1%size()) / coef%volume
    end if
    call field_col3(w1, omega_x, omega_x)
    call field_addcol3(w1, omega_y, omega_y)
    call field_addcol3(w1, omega_z, omega_z)
    if (NEKO_BCKND_DEVICE .eq. 1) then

       e2 = 0.5_rp * device_glsc2(w1%x_d, coef%B_d, w1%size()) / coef%volume
    else
       e2 = 0.5_rp * glsc2(w1%x, coef%B, w1%size()) / coef%volume
    end if

    if (pe_rank .eq. 0) then
       write(*,'(a,e18.9,a,e18.9,a,e18.9)') &
            'POST: t:', time%t, ' Ekin:', e1, ' enst:', e2
    end if

    call omega_variance(time)

  end subroutine user_calc_quantities

  ! Compute and print the variance of the vorticity vector over the domain.
  ! Variance is defined as <|omega|^2> - |<omega>|^2, where the averages are
  ! taken with the mass-matrix weighted volume average used elsewhere in Neko
  ! (e.g. for kinetic energy and enstrophy).
  subroutine omega_variance(time)
    type(time_state_t), intent(in) :: time

    type(field_t), pointer :: omega_x, omega_y, omega_z, u
    type(field_t) :: w_sq
    real(kind=rp) :: vol, mean_wx, mean_wy, mean_wz
    real(kind=rp) :: mean_w2, var_w
    type(coef_t), pointer :: coef

    coef => neko_user_access%case%fluid%c_Xh

    u => neko_registry%get_field('u')
    omega_x => neko_registry%get_field("omega_x")
    omega_y => neko_registry%get_field("omega_y")
    omega_z => neko_registry%get_field("omega_z")

    ! Work array for the squared vorticity magnitude
    call w_sq%init(u%dof, 'w_sq')

    ! Compute |omega|^2 = omega_x^2 + omega_y^2 + omega_z^2 point-wise
    call field_col3(w_sq, omega_x, omega_x)    ! w_sq = omega_x * omega_x
    call field_addcol3(w_sq, omega_y, omega_y) ! w_sq = w_sq + omega_y * omega_y
    call field_addcol3(w_sq, omega_z, omega_z) ! w_sq = w_sq + omega_z * omega_z

    ! Domain volume (using the mass matrix / quadrature weights)
    vol = coef%volume

    ! Mass-weighted mean of each vorticity component
    ! <f> = integral(f) / volume = glsc2(f, B) / volume
    if (NEKO_BCKND_DEVICE .eq. 1) then
       mean_wx = device_glsc2(omega_x%x_d, coef%B_d, omega_x%size()) / vol
       mean_wy = device_glsc2(omega_y%x_d, coef%B_d, omega_y%size()) / vol
       mean_wz = device_glsc2(omega_z%x_d, coef%B_d, omega_z%size()) / vol
       mean_w2 = device_glsc2(w_sq%x_d, coef%B_d, w_sq%size()) / vol
    else
       mean_wx = glsc2(omega_x%x, coef%B, omega_x%size()) / vol
       mean_wy = glsc2(omega_y%x, coef%B, omega_y%size()) / vol
       mean_wz = glsc2(omega_z%x, coef%B, omega_z%size()) / vol
       mean_w2 = glsc2(w_sq%x, coef%B, w_sq%size()) / vol
    end if

    ! Variance of the vorticity vector
    var_w = mean_w2 - (mean_wx**2 + mean_wy**2 + mean_wz**2)

    if (pe_rank .eq. 0) then
       write(*,'(a,e18.9,a,e18.9)') &
            'VARIANCE: t:', time%t, ' var:', var_w
    end if

    call w_sq%free()

  end subroutine omega_variance

  ! User-defined finalization routine called at the end of the simulation
  subroutine user_finalize(time)
    type(time_state_t), intent(in) :: time

    ! Deallocate the fields
    call w1%free()
  end subroutine user_finalize

end module user
