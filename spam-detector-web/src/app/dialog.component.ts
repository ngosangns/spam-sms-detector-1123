import { CommonModule } from '@angular/common';
import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';

@Component({
  standalone: true,
  imports: [CommonModule],
  selector: 'app-dialog',
  template: `
    <div class="p-6">
      <p mat-dialog-title class="font-bold text-lg">Kết quả</p>
      <hr />
      <div mat-dialog-content class="mt-4">
        <p>
          Phân loại:
          <span class="font-bold" [ngStyle]="{ color: getColor(data.percent) }">
            {{ data.type === 'ham' ? 'Thông thường' : 'Spam' }}
          </span>
        </p>
        <p>
          Tỉ lệ spam:
          <span class="font-bold" [ngStyle]="{ color: getColor(data.percent) }">
            {{ (data.percent * 100).toFixed(2) }}%
          </span>
        </p>
      </div>
    </div>
  `,
})
export class DialogComponent {
  constructor(
    @Inject(MAT_DIALOG_DATA) public data: { type: string; percent: number }
  ) {}

  getColor(percent: number): string {
    if (percent > 0.75) {
      return 'red';
    } else if (percent > 0.5) {
      return 'orange';
    } else if (percent > 0.25) {
      return 'yellow';
    } else {
      return 'green';
    }
  }
}
